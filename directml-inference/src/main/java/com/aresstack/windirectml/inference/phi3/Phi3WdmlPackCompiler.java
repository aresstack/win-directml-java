package com.aresstack.windirectml.inference.phi3;

import com.aresstack.windirectml.inference.model.WdmlPackWriter;
import com.aresstack.windirectml.inference.model.WdmlPackWriter.PayloadWriter;
import com.aresstack.windirectml.inference.phi3.Phi3Weights.LayerWeights;
import com.aresstack.windirectml.inference.phi3.Phi3Weights.QuantizedWeight;
import com.aresstack.windirectml.inference.phi3.Phi3WeightLayout.ExtRef;
import com.aresstack.windirectml.inference.phi3.Phi3WeightLayout.LayerRef;
import com.aresstack.windirectml.inference.phi3.Phi3WeightLayout.QuantRef;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Phi-3 ONNX&rarr;wdmlpack compiler (PHI3-WDMLPACK-COMPILER-1/2).
 *
 * <p>Serializes the exact tensors the {@link Phi3Runtime} consumes into a payload-carrying
 * {@code model_phi3.wdmlpack}: INT4 MatMulNBits projections as {@code qweight/scales/zeropoints} triplets (byte-exact)
 * and the fp32 norm/embedding/cos-sin vectors. {@link Phi3RuntimePackage} reads it back and reconstructs
 * {@code Phi3Weights}, so the runtime stays unchanged.</p>
 *
 * <p><b>Heap-safe (COMPILER-2):</b> {@link #compile} streams each tensor straight from the mmap'd
 * {@code model.onnx.data} via {@link Phi3Weights#planLayout} — it never materializes the full ~2.4 GB model on the
 * heap, so the real Phi-3-mini compile runs within a small heap. The in-memory {@link #writePackage} core is shared
 * (it backs the synthetic round-trip test).</p>
 *
 * <p>This builds the compiler + package only. It does <b>not</b> wire a Workbench lifecycle/gate, so Phi-3 stays
 * {@code PLANNED} / not runnable in the Workbench.</p>
 */
public final class Phi3WdmlPackCompiler {

    public static final String ARCHITECTURE = "phi3";
    public static final int COMPILER_VERSION = 2;
    static final int PAYLOAD_TENSOR_ALIGNMENT = 64;
    private static final int STREAM_CHUNK = 1 << 16;

    private static final Logger log = LoggerFactory.getLogger(Phi3WdmlPackCompiler.class);

    private Phi3WdmlPackCompiler() {
    }

    /** Result of a compile. */
    public record Phi3CompileResult(Path output, int tensorCount, long payloadBytes, int layers) {
    }

    // ── Streaming ONNX compile (heap-safe) ───────────────────────────────

    /** Compile a Phi-3 ONNX model directory into a {@code .wdmlpack}, streaming tensors from the source mmap. */
    public static Phi3CompileResult compile(Phi3CompileOptions options) throws IOException {
        Path output = options.resolveOutput();
        if (Files.exists(output) && !options.force()) {
            throw new IOException("Phi-3 package already exists (use force): " + output);
        }
        Path configPath = options.modelDir().resolve("config.json");
        if (!Files.isRegularFile(configPath)) {
            throw new IOException("Missing Phi-3 config.json in " + options.modelDir());
        }
        Phi3Config config = Phi3Config.load(configPath);
        Phi3WeightLayout layout = Phi3Weights.planLayout(options.modelDir(), config);
        boolean tokenizerPresent = Files.isRegularFile(options.modelDir().resolve("tokenizer.json"));

        long dataSize = Files.size(layout.externalDataPath());
        if (dataSize > Integer.MAX_VALUE) {
            throw new IOException("Phi-3 model.onnx.data too large for the current mmap reader: " + dataSize);
        }
        try (FileChannel channel = FileChannel.open(layout.externalDataPath(), StandardOpenOption.READ)) {
            MappedByteBuffer src = channel.map(FileChannel.MapMode.READ_ONLY, 0, dataSize);
            src.order(ByteOrder.LITTLE_ENDIAN);

            List<TensorSource> sources = new ArrayList<>();
            sources.add(fp16Source(Phi3WdmlPackRoles.EMBED_TOKENS, src, layout.embedTokens(),
                    new long[]{config.vocabSize(), config.hiddenSize()}));
            sources.add(floatSource(Phi3WdmlPackRoles.COS_CACHE, layout.cosCache(),
                    new long[]{layout.cosCache().length}));
            sources.add(floatSource(Phi3WdmlPackRoles.SIN_CACHE, layout.sinCache(),
                    new long[]{layout.sinCache().length}));
            sources.add(fp16Source(Phi3WdmlPackRoles.FINAL_NORM, src, layout.finalNorm(),
                    new long[]{fp16Count(layout.finalNorm())}));
            addQuantStream(sources, Phi3WdmlPackRoles.LM_HEAD, src, layout.lmHead());

            LayerRef[] layers = layout.layers();
            for (int l = 0; l < layers.length; l++) {
                LayerRef lr = layers[l];
                sources.add(fp16Source(Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.INPUT_NORM), src, lr.inputNorm(),
                        new long[]{fp16Count(lr.inputNorm())}));
                sources.add(fp16Source(Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.POST_NORM), src, lr.postNorm(),
                        new long[]{fp16Count(lr.postNorm())}));
                sources.add(floatSource(Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.ATTN_OUT_SCALE),
                        lr.attnOutScale(), new long[]{lr.attnOutScale().length}));
                sources.add(floatSource(Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.MLP_OUT_SCALE),
                        lr.mlpOutScale(), new long[]{lr.mlpOutScale().length}));
                addQuantStream(sources, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.Q_PROJ), src, lr.qProj());
                addQuantStream(sources, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.K_PROJ), src, lr.kProj());
                addQuantStream(sources, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.V_PROJ), src, lr.vProj());
                addQuantStream(sources, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.O_PROJ), src, lr.oProj());
                addQuantStream(sources, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.GATE_UP_PROJ), src, lr.gateUpProj());
                addQuantStream(sources, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.DOWN_PROJ), src, lr.downProj());
            }
            return writeContainer(config, sources, output, tokenizerPresent, layers.length);
        }
    }

    // ── In-memory compile (synthetic round-trip / small callers) ─────────

    /** Serialize already-extracted {@link Phi3Weights} into a {@code .wdmlpack} (no ONNX dependency). */
    public static Phi3CompileResult writePackage(Phi3Config config, Phi3Weights weights, Path output,
                                                 boolean tokenizerPresent) throws IOException {
        List<TensorSource> sources = new ArrayList<>();
        sources.add(floatSource(Phi3WdmlPackRoles.EMBED_TOKENS, weights.embedTokens,
                new long[]{config.vocabSize(), config.hiddenSize()}));
        sources.add(floatSource(Phi3WdmlPackRoles.COS_CACHE, weights.cosCache, new long[]{weights.cosCache.length}));
        sources.add(floatSource(Phi3WdmlPackRoles.SIN_CACHE, weights.sinCache, new long[]{weights.sinCache.length}));
        sources.add(floatSource(Phi3WdmlPackRoles.FINAL_NORM, weights.finalNormWeight,
                new long[]{weights.finalNormWeight.length}));
        addQuantInMemory(sources, Phi3WdmlPackRoles.LM_HEAD, weights.lmHead);

        LayerWeights[] layers = weights.layers;
        for (int l = 0; l < layers.length; l++) {
            LayerWeights lw = layers[l];
            sources.add(floatSource(Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.INPUT_NORM), lw.inputNormWeight(),
                    new long[]{lw.inputNormWeight().length}));
            sources.add(floatSource(Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.POST_NORM), lw.postNormWeight(),
                    new long[]{lw.postNormWeight().length}));
            sources.add(floatSource(Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.ATTN_OUT_SCALE), lw.attnOutScale(),
                    new long[]{lw.attnOutScale().length}));
            sources.add(floatSource(Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.MLP_OUT_SCALE), lw.mlpOutScale(),
                    new long[]{lw.mlpOutScale().length}));
            addQuantInMemory(sources, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.Q_PROJ), lw.qProj());
            addQuantInMemory(sources, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.K_PROJ), lw.kProj());
            addQuantInMemory(sources, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.V_PROJ), lw.vProj());
            addQuantInMemory(sources, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.O_PROJ), lw.oProj());
            addQuantInMemory(sources, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.GATE_UP_PROJ), lw.gateUpProj());
            addQuantInMemory(sources, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.DOWN_PROJ), lw.downProj());
        }
        return writeContainer(config, sources, output, tokenizerPresent, layers.length);
    }

    // ── Shared serialization core ────────────────────────────────────────

    private static Phi3CompileResult writeContainer(Phi3Config config, List<TensorSource> sources, Path output,
                                                    boolean tokenizerPresent, int layers) throws IOException {
        List<WdmlPackWriter.PayloadEntry> entries = new ArrayList<>(sources.size());
        List<Map<String, Object>> tensorDir = new ArrayList<>(sources.size());
        long cursor = 0;
        for (TensorSource s : sources) {
            cursor = WdmlPackWriter.align(cursor, PAYLOAD_TENSOR_ALIGNMENT);
            long offset = cursor;
            entries.add(new WdmlPackWriter.PayloadEntry(s.name, offset, s.byteLength, s.writer));
            Map<String, Object> t = new LinkedHashMap<>();
            t.put("name", s.name);
            t.put("dataType", s.dataType);
            t.put("dims", toList(s.dims));
            t.put("byteLength", s.byteLength);
            t.put("payloadOffset", offset);
            t.put("payloadLength", s.byteLength);
            tensorDir.add(t);
            cursor += s.byteLength;
        }
        Map<String, Object> manifest = buildManifest(config, tensorDir, cursor, tokenizerPresent);
        WdmlPackWriter.writeWithPayload(output, manifest, entries, cursor);
        log.info("Wrote Phi-3 wdmlpack: {} (tensors={}, payload={} bytes, layers={})",
                output, sources.size(), cursor, layers);
        return new Phi3CompileResult(output, sources.size(), cursor, layers);
    }

    private static Map<String, Object> buildManifest(Phi3Config config, List<Map<String, Object>> tensorDir,
                                                     long payloadBytes, boolean tokenizerPresent) {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("format", "wdmlpack");
        root.put("version", WdmlPackWriter.VERSION);
        root.put("createdAt", Instant.now().toString());
        root.put("architecture", ARCHITECTURE);
        root.put("mode", "payload");
        root.put("payloadIncluded", true);
        root.put("runtimeLoadable", true);
        root.put("runtimeLoadMode", "phi3-native-payload");
        root.put("payloadAlignment", WdmlPackWriter.PAYLOAD_ALIGNMENT);
        root.put("payloadBytes", payloadBytes);
        root.put("tokenizer", "tokenizer.json");
        root.put("tokenizerPresentAtCompile", tokenizerPresent);

        Map<String, Object> model = new LinkedHashMap<>();
        model.put("hiddenSize", config.hiddenSize());
        model.put("numAttentionHeads", config.numAttentionHeads());
        model.put("numHiddenLayers", config.numHiddenLayers());
        model.put("numKeyValueHeads", config.numKeyValueHeads());
        model.put("vocabSize", config.vocabSize());
        model.put("maxPositionEmbeddings", config.maxPositionEmbeddings());
        model.put("intermediateSize", config.intermediateSize());
        model.put("rmsNormEps", config.rmsNormEps());
        model.put("ropeTheta", config.ropeTheta());
        root.put("model", model);

        Map<String, Object> cache = new LinkedHashMap<>();
        cache.put("compiler", "Phi3WdmlPackCompiler");
        cache.put("compilerVersion", COMPILER_VERSION);
        cache.put("createdBy", "win-directml-java");
        root.put("cache", cache);

        root.put("tensors", tensorDir);
        return root;
    }

    // ── Source builders ──────────────────────────────────────────────────

    private static TensorSource floatSource(String name, float[] data, long[] dims) {
        return new TensorSource(name, Phi3WdmlPackRoles.DTYPE_FLOAT, dims,
                (long) data.length * Float.BYTES, channel -> writeFloats(channel, data));
    }

    private static TensorSource fp16Source(String name, ByteBuffer src, ExtRef ref, long[] dims) {
        long byteLength = fp16Count(ref) * (long) Float.BYTES;
        return new TensorSource(name, Phi3WdmlPackRoles.DTYPE_FLOAT, dims, byteLength,
                channel -> streamFp16ToFp32(channel, src, ref.offset(), ref.length()));
    }

    private static TensorSource rawSource(String name, int dataType, ByteBuffer src, ExtRef ref, long[] dims) {
        return new TensorSource(name, dataType, dims, ref.length(),
                channel -> streamRaw(channel, src, ref.offset(), ref.length()));
    }

    private static void addQuantStream(List<TensorSource> sources, String role, ByteBuffer src, QuantRef q) {
        sources.add(rawSource(Phi3WdmlPackRoles.qweight(role), Phi3WdmlPackRoles.DTYPE_UINT8, src, q.qData(),
                new long[]{q.N(), q.K(), q.blockSize()}));
        long scaleCount = fp16Count(q.scales());
        sources.add(new TensorSource(Phi3WdmlPackRoles.scales(role), Phi3WdmlPackRoles.DTYPE_FLOAT,
                new long[]{scaleCount}, scaleCount * Float.BYTES,
                channel -> streamFp16ToFp32(channel, src, q.scales().offset(), q.scales().length())));
        if (q.zeroPoints() != null) {
            sources.add(rawSource(Phi3WdmlPackRoles.zeropoints(role), Phi3WdmlPackRoles.DTYPE_UINT8, src,
                    q.zeroPoints(), new long[]{q.zeroPoints().length()}));
        } else {
            // Default zero point = 8 for 4-bit (packed 0x88), matching Phi3Weights.loadQuantizedWeight.
            int numBlocks = (int) scaleCount;
            byte[] fill = new byte[(numBlocks + 1) / 2];
            Arrays.fill(fill, (byte) 0x88);
            sources.add(new TensorSource(Phi3WdmlPackRoles.zeropoints(role), Phi3WdmlPackRoles.DTYPE_UINT8,
                    new long[]{fill.length}, fill.length, channel -> writeBytes(channel, fill)));
        }
    }

    private static void addQuantInMemory(List<TensorSource> sources, String role, QuantizedWeight w) {
        // Reference the transient byte arrays directly (no clone): the source records are discarded after compile.
        sources.add(new TensorSource(Phi3WdmlPackRoles.qweight(role), Phi3WdmlPackRoles.DTYPE_UINT8,
                new long[]{w.N(), w.K(), w.blockSize()}, w.qWeight().length, channel -> writeBytes(channel, w.qWeight())));
        sources.add(floatSource(Phi3WdmlPackRoles.scales(role), w.scales(), new long[]{w.scales().length}));
        sources.add(new TensorSource(Phi3WdmlPackRoles.zeropoints(role), Phi3WdmlPackRoles.DTYPE_UINT8,
                new long[]{w.zeroPoints().length}, w.zeroPoints().length,
                channel -> writeBytes(channel, w.zeroPoints())));
    }

    private static long fp16Count(ExtRef ref) {
        return ref.length() / 2;
    }

    // ── Payload writers ──────────────────────────────────────────────────

    private static void streamFp16ToFp32(FileChannel channel, ByteBuffer src, long offset, long fp16ByteLen)
            throws IOException {
        ByteBuffer in = src.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        in.position(Math.toIntExact(offset));
        in.limit(Math.toIntExact(offset + fp16ByteLen));
        int count = Math.toIntExact(fp16ByteLen / 2);
        ByteBuffer out = ByteBuffer.allocate(STREAM_CHUNK).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            if (out.remaining() < Float.BYTES) {
                flush(channel, out);
            }
            out.putFloat(Phi3Weights.fp16ToFp32(in.getShort()));
        }
        flush(channel, out);
    }

    private static void streamRaw(FileChannel channel, ByteBuffer src, long offset, long len) throws IOException {
        ByteBuffer in = src.duplicate();
        in.position(Math.toIntExact(offset));
        in.limit(Math.toIntExact(offset + len));
        ByteBuffer slice = in.slice();
        while (slice.hasRemaining()) {
            channel.write(slice);
        }
    }

    private static void writeFloats(FileChannel channel, float[] data) throws IOException {
        ByteBuffer out = ByteBuffer.allocate(STREAM_CHUNK).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : data) {
            if (out.remaining() < Float.BYTES) {
                flush(channel, out);
            }
            out.putFloat(v);
        }
        flush(channel, out);
    }

    private static void writeBytes(FileChannel channel, byte[] data) throws IOException {
        ByteBuffer buf = ByteBuffer.wrap(data);
        while (buf.hasRemaining()) {
            channel.write(buf);
        }
    }

    private static void flush(FileChannel channel, ByteBuffer out) throws IOException {
        out.flip();
        while (out.hasRemaining()) {
            channel.write(out);
        }
        out.clear();
    }

    private static List<Long> toList(long[] dims) {
        List<Long> out = new ArrayList<>(dims.length);
        for (long d : dims) {
            out.add(d);
        }
        return out;
    }

    private record TensorSource(String name, int dataType, long[] dims, long byteLength, PayloadWriter writer) {
    }
}
