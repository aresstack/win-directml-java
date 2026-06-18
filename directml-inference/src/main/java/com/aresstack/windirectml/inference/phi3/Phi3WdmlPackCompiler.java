package com.aresstack.windirectml.inference.phi3;

import com.aresstack.windirectml.inference.model.WdmlPackWriter;
import com.aresstack.windirectml.inference.phi3.Phi3Weights.LayerWeights;
import com.aresstack.windirectml.inference.phi3.Phi3Weights.QuantizedWeight;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal Phi-3 ONNX&rarr;wdmlpack compiler (PHI3-WDMLPACK-COMPILER-1).
 *
 * <p>Extracts {@link Phi3Weights} from the Phi-3 ONNX source (via {@link Phi3OnnxModelSource}, which reuses the
 * existing graph-topology extraction) and serializes the exact tensors the {@link Phi3Runtime} consumes into a
 * payload-carrying {@code model_phi3.wdmlpack}: INT4 MatMulNBits projections as {@code qweight/scales/zeropoints}
 * triplets (lossless byte round-trip) and the fp32 norm/embedding/cos-sin vectors. The package is read back by
 * {@link Phi3RuntimePackage}, which reconstructs {@code Phi3Weights} so the runtime stays unchanged.</p>
 *
 * <p>This slice builds the compiler + package foundation only. It does <b>not</b> wire a Workbench lifecycle/gate, so
 * Phi-3 stays {@code PLANNED} / not runnable in the Workbench.</p>
 */
public final class Phi3WdmlPackCompiler {

    public static final String ARCHITECTURE = "phi3";
    public static final int COMPILER_VERSION = 1;
    static final int PAYLOAD_TENSOR_ALIGNMENT = 64;

    private static final Logger log = LoggerFactory.getLogger(Phi3WdmlPackCompiler.class);

    private Phi3WdmlPackCompiler() {
    }

    /** Result of a compile. */
    public record Phi3CompileResult(Path output, int tensorCount, long payloadBytes, int layers) {
    }

    /** Compile a Phi-3 ONNX model directory into a {@code .wdmlpack}. */
    public static Phi3CompileResult compile(Phi3CompileOptions options) throws IOException {
        Path output = options.resolveOutput();
        if (Files.exists(output) && !options.force()) {
            throw new IOException("Phi-3 package already exists (use force): " + output);
        }
        Phi3OnnxModelSource source = new Phi3OnnxModelSource(options.modelDir());
        Phi3OnnxModelSource.Imported imported = source.load();
        try {
            return writePackage(imported.config(), imported.weights(), output, imported.tokenizerPresent());
        } finally {
            imported.weights().close();
        }
    }

    /**
     * Serialize already-extracted {@link Phi3Weights} into a {@code .wdmlpack}. This is the testable core: it has no
     * ONNX dependency, so a synthetic {@code Phi3Weights} (see {@link Phi3Weights#ofRecords}) round-trips through it.
     */
    public static Phi3CompileResult writePackage(Phi3Config config, Phi3Weights weights, Path output,
                                                 boolean tokenizerPresent) throws IOException {
        List<TensorSpec> specs = new ArrayList<>();
        addFloat(specs, Phi3WdmlPackRoles.EMBED_TOKENS, weights.embedTokens,
                new long[]{config.vocabSize(), config.hiddenSize()});
        addFloat(specs, Phi3WdmlPackRoles.COS_CACHE, weights.cosCache, new long[]{weights.cosCache.length});
        addFloat(specs, Phi3WdmlPackRoles.SIN_CACHE, weights.sinCache, new long[]{weights.sinCache.length});
        addFloat(specs, Phi3WdmlPackRoles.FINAL_NORM, weights.finalNormWeight,
                new long[]{weights.finalNormWeight.length});
        addQuantized(specs, Phi3WdmlPackRoles.LM_HEAD, weights.lmHead);

        LayerWeights[] layers = weights.layers;
        for (int l = 0; l < layers.length; l++) {
            LayerWeights lw = layers[l];
            addFloat(specs, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.INPUT_NORM), lw.inputNormWeight(),
                    new long[]{lw.inputNormWeight().length});
            addFloat(specs, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.POST_NORM), lw.postNormWeight(),
                    new long[]{lw.postNormWeight().length});
            addFloat(specs, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.ATTN_OUT_SCALE), lw.attnOutScale(),
                    new long[]{lw.attnOutScale().length});
            addFloat(specs, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.MLP_OUT_SCALE), lw.mlpOutScale(),
                    new long[]{lw.mlpOutScale().length});
            addQuantized(specs, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.Q_PROJ), lw.qProj());
            addQuantized(specs, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.K_PROJ), lw.kProj());
            addQuantized(specs, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.V_PROJ), lw.vProj());
            addQuantized(specs, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.O_PROJ), lw.oProj());
            addQuantized(specs, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.GATE_UP_PROJ), lw.gateUpProj());
            addQuantized(specs, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.DOWN_PROJ), lw.downProj());
        }

        // Lay out the payload (aligned), build the manifest tensor directory.
        List<WdmlPackWriter.PayloadEntry> entries = new ArrayList<>(specs.size());
        List<Map<String, Object>> tensorDir = new ArrayList<>(specs.size());
        long cursor = 0;
        for (TensorSpec spec : specs) {
            cursor = WdmlPackWriter.align(cursor, PAYLOAD_TENSOR_ALIGNMENT);
            long offset = cursor;
            byte[] data = spec.data;
            entries.add(new WdmlPackWriter.PayloadEntry(spec.name, offset, data.length,
                    channel -> writeAll(channel, data)));
            Map<String, Object> t = new LinkedHashMap<>();
            t.put("name", spec.name);
            t.put("dataType", spec.dataType);
            t.put("dims", toList(spec.dims));
            t.put("byteLength", (long) data.length);
            t.put("payloadOffset", offset);
            t.put("payloadLength", (long) data.length);
            tensorDir.add(t);
            cursor += data.length;
        }

        Map<String, Object> manifest = buildManifest(config, tensorDir, cursor, tokenizerPresent);
        WdmlPackWriter.writeWithPayload(output, manifest, entries, cursor);
        log.info("Wrote Phi-3 wdmlpack: {} (tensors={}, payload={} bytes, layers={})",
                output, specs.size(), cursor, layers.length);
        return new Phi3CompileResult(output, specs.size(), cursor, layers.length);
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
        // Tokenizer is an inference-time sibling asset (tokenizer.json), referenced not embedded -- same convention
        // as the other families. The package records whether it was present at compile time.
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

    private static void addFloat(List<TensorSpec> specs, String name, float[] data, long[] dims) {
        ByteBuffer buf = ByteBuffer.allocate(data.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : data) {
            buf.putFloat(v);
        }
        specs.add(new TensorSpec(name, Phi3WdmlPackRoles.DTYPE_FLOAT, dims, buf.array()));
    }

    private static void addQuantized(List<TensorSpec> specs, String role, QuantizedWeight w) {
        // qweight dims encode [N, K, blockSize] so the loader can rebuild QuantizedWeight without extra metadata.
        // The source records are transient (discarded after compile), so we reference their byte arrays directly
        // rather than cloning -- halving the peak INT4 footprint on the heavy real-model path.
        specs.add(new TensorSpec(Phi3WdmlPackRoles.qweight(role), Phi3WdmlPackRoles.DTYPE_UINT8,
                new long[]{w.N(), w.K(), w.blockSize()}, w.qWeight()));
        ByteBuffer scales = ByteBuffer.allocate(w.scales().length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float s : w.scales()) {
            scales.putFloat(s);
        }
        specs.add(new TensorSpec(Phi3WdmlPackRoles.scales(role), Phi3WdmlPackRoles.DTYPE_FLOAT,
                new long[]{w.scales().length}, scales.array()));
        specs.add(new TensorSpec(Phi3WdmlPackRoles.zeropoints(role), Phi3WdmlPackRoles.DTYPE_UINT8,
                new long[]{w.zeroPoints().length}, w.zeroPoints()));
    }

    private static void writeAll(java.nio.channels.FileChannel channel, byte[] data) throws IOException {
        ByteBuffer buf = ByteBuffer.wrap(data);
        while (buf.hasRemaining()) {
            channel.write(buf);
        }
    }

    private static List<Long> toList(long[] dims) {
        List<Long> out = new ArrayList<>(dims.length);
        for (long d : dims) {
            out.add(d);
        }
        return out;
    }

    private record TensorSpec(String name, int dataType, long[] dims, byte[] data) {
    }
}
