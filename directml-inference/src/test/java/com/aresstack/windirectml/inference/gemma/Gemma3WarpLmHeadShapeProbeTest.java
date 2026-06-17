package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorEntry;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorsFile;
import com.aresstack.windirectml.inference.warp.WarpDenseProjection;
import com.aresstack.windirectml.windows.WarpExecutionContext;
import com.aresstack.windirectml.windows.WarpGpuBuffer;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-18: probe DML-GEMM shapes for the tied Gemma LM head (the 262144×640 matvec that is ~52% of
 * the WARP decode token, per GEMMA-WARP-17). Gated/heavy:
 * {@code -Dgemma.warp.realModel=true -Dgemma.warp.lmHeadProbe=true}. Measured on the explicit WARP software
 * adapter (the CPU-only product path).
 *
 * <p>This slice only <b>measures</b> — it does not change the product LM-head path. Variants compared: the
 * current GEMV ({@code matvecResident}), the GEMM M=1 form ({@code matmulBatchResident(1)}), and row-blocked
 * splits of the vocab into several smaller DML-GEMMs (coalesced). DML-GEMM is kept throughout; no INT4, no
 * custom kernel, no {@code .wdmlpack} change. Correctness: every variant must produce the same full-vocab
 * Top-1 as the baseline, and the product path still yields " Paris" (token 9079).</p>
 */
@EnabledOnOs(OS.WINDOWS)
@EnabledIfSystemProperty(named = "gemma.warp.realModel", matches = "true")
class Gemma3WarpLmHeadShapeProbeTest {

    private static final int[] FRANCE_IDS = {2, 818, 5279, 529, 7001, 563};
    private static final int EXPECTED_NEXT = 9079; // " Paris"
    private static final int WARM_ITERS = 2;
    private static final int MEASURE_ITERS = 5;
    private static final int REF_DECODE_TOKENS = 6;

    private static WindowsBindings wb;

    @BeforeAll
    static void initGpu() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML device");
        wb = new WindowsBindings();
        wb.init("directml", WindowsBindings.AdapterMode.WARP); // WARP-focused: the CPU-only product path
    }

    @AfterAll
    static void closeGpu() {
        if (wb != null) {
            wb.close();
        }
    }

    private record Variant(String name, String layout, int dispatches, double msPerToken, int top1, double memMb) {
    }

    @Test
    void lmHeadShapeProbe() throws Exception {
        assumeTrue(Boolean.getBoolean("gemma.warp.lmHeadProbe"), "opt-in via -Dgemma.warp.lmHeadProbe=true");
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");

        Gemma3Config config = new Gemma3ConfigReader().read(dir.resolve("config.json"));
        int vocab = config.vocabSize();
        int hidden = config.hiddenSize();
        SafeTensorsFile file = SafeTensorsReader.read(dir.resolve("model.safetensors"));
        ByteBuffer embBb = decodeEmbeddingDirectFp32(file.tensors().get(Gemma3TensorNameMapper.EMBED_TOKENS));

        // ── Paris smoke + real decode avg/token reference (product path, unchanged) ──
        double realDecodeMsPerToken;
        {
            float[] finalNorm = Gemma3ReferenceWeights.decodeFloats(file.tensors().get(Gemma3TensorNameMapper.FINAL_NORM));
            Gemma3WarpLayerWeights[] layers = loadWarpLayers(file, config);
            Gemma3WarpWeights weights = Gemma3WarpWeights.ofByteBufferEmbedding(config, embBb, finalNorm, layers);
            try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, weights)) {
                int tok = DecoderOnlyMath.argmax(sess.prefillResidentBatched(FRANCE_IDS));
                assertEquals(EXPECTED_NEXT, tok, "WARP product LM head must give \" Paris\"");
                for (int i = 0; i < 2; i++) { // warm
                    tok = sess.decodeNextTokenResident(tok);
                }
                long t = System.nanoTime();
                for (int i = 0; i < REF_DECODE_TOKENS; i++) {
                    tok = sess.decodeNextTokenResident(tok);
                }
                realDecodeMsPerToken = (System.nanoTime() - t) / 1e6 / REF_DECODE_TOKENS;
            }
        }

        // ── LM-head shape variants over a fixed input vector ──
        WarpExecutionContext ctx = new WarpExecutionContext(wb);
        float[] x = randomVector(new Random(18), hidden);
        double weightMb = (double) vocab * hidden * Float.BYTES / (1024 * 1024);
        List<Variant> variants = new ArrayList<>();
        double baselineMs;
        float[] baseLogits = new float[vocab];

        try (WarpGpuBuffer xBuf = WarpGpuBuffer.upload(wb, x)) {
            // Baseline GEMV + GEMM(M=1) share one full projection.
            try (WarpDenseProjection full = WarpDenseProjection.fromDequantizedWeights(
                    wb, "lmhead.full", vocab, hidden, embBb.duplicate().order(ByteOrder.LITTLE_ENDIAN))) {
                baselineMs = measure(ctx, xBuf, List.of(full), false, vocab, baseLogits);
                int baseTop1 = DecoderOnlyMath.argmax(baseLogits);
                variants.add(new Variant("baseline-matvec (GEMV)", "[" + vocab + "," + hidden + "]·[" + hidden + "]",
                        1, baselineMs, baseTop1, weightMb));

                float[] gemmLogits = new float[vocab];
                double gemmMs = measure(ctx, xBuf, List.of(full), true, vocab, gemmLogits);
                variants.add(new Variant("batched-m1 (GEMM M=1)", "[1," + hidden + "]·[" + vocab + "," + hidden + "]ᵀ",
                        1, gemmMs, DecoderOnlyMath.argmax(gemmLogits), weightMb));
                assertCloseTop1("batched-m1", baseLogits, gemmLogits);
            }

            // Row-blocked variants: split the vocab into B contiguous row blocks (same total weights).
            for (int blocks : new int[]{4, 16}) {
                List<WarpDenseProjection> parts = buildRowBlocks(embBb, vocab, hidden, blocks);
                try {
                    float[] mvLogits = new float[vocab];
                    double mvMs = measure(ctx, xBuf, parts, false, vocab, mvLogits);
                    variants.add(new Variant("rowblock-" + blocks + "-matvec", blocks + "×[" + (vocab / blocks) + ","
                            + hidden + "] coalesced", blocks, mvMs, DecoderOnlyMath.argmax(mvLogits), weightMb));
                    assertCloseTop1("rowblock-" + blocks + "-matvec", baseLogits, mvLogits);
                } finally {
                    parts.forEach(WarpDenseProjection::close);
                }
            }
        }
        int baselineTop1 = DecoderOnlyMath.argmax(baseLogits);

        // ── Report ──
        double restOfDecodeMs = Math.max(0, realDecodeMsPerToken - baselineMs);
        System.out.println("===== [GEMMA-WARP-18] LM-head DML-GEMM shape probe (WARP) =====");
        System.out.printf(Locale.ROOT, "real decode avg/token=%.1f ms; baseline lm-head=%.1f ms; "
                + "rest-of-decode=%.1f ms; output shape=[%d]%n", realDecodeMsPerToken, baselineMs, restOfDecodeMs, vocab);
        System.out.printf(Locale.ROOT, "  %-26s %-30s %9s %12s %14s %9s %8s%n",
                "variant", "layout", "dispatch", "lmhead ms", "decode est ms", "mem MB", "top1");
        for (Variant v : variants) {
            System.out.printf(Locale.ROOT, "  %-26s %-30s %9d %12.1f %14.1f %9.0f %8d%n",
                    v.name(), v.layout(), v.dispatches(), v.msPerToken(), restOfDecodeMs + v.msPerToken(),
                    v.memMb(), v.top1());
            assertEquals(baselineTop1, v.top1(), "variant " + v.name() + " must keep the baseline Top-1");
        }
    }

    /**
     * Measure the matmul-only ms/token of an LM-head variant (warm + averaged), filling {@code logitsOut} on
     * the last iteration. {@code batched} selects {@code matmulBatchResident(1)} (synchronous per block) over
     * the GEMV; matvec blocks are coalesced into one submit and flushed (one synchronous fence).
     */
    private double measure(WarpExecutionContext ctx, WarpGpuBuffer xBuf, List<WarpDenseProjection> blocks,
                           boolean batched, int vocab, float[] logitsOut) throws Exception {
        return measure(ctx, xBuf, blocks, batched, vocab, MEASURE_ITERS, logitsOut);
    }

    private double measure(WarpExecutionContext ctx, WarpGpuBuffer xBuf, List<WarpDenseProjection> blocks,
                           boolean batched, int vocab, int iters, float[] logitsOut) throws Exception {
        for (int w = 0; w < WARM_ITERS; w++) {
            runOnce(ctx, xBuf, blocks, batched, vocab, null);
        }
        long total = 0;
        for (int i = 0; i < iters; i++) {
            float[] sink = (i == iters - 1) ? logitsOut : null;
            total += runOnce(ctx, xBuf, blocks, batched, vocab, sink);
        }
        return total / 1e6 / iters;
    }

    /** One LM-head evaluation; returns the matmul-region nanoseconds. Fills {@code logitsOut} when non-null. */
    private long runOnce(WarpExecutionContext ctx, WarpGpuBuffer xBuf, List<WarpDenseProjection> blocks,
                         boolean batched, int vocab, float[] logitsOut) throws Exception {
        List<WarpGpuBuffer> outs = new ArrayList<>(blocks.size());
        long nanos;
        try {
            long t0 = System.nanoTime();
            if (batched) {
                for (WarpDenseProjection b : blocks) {
                    outs.add(b.forwardResidentBatched(ctx, xBuf, 1)); // GEMM M=1, synchronous
                }
            } else {
                ctx.beginRecording();
                for (WarpDenseProjection b : blocks) {
                    outs.add(b.forwardResident(ctx, xBuf)); // GEMV, recorded
                }
                ctx.flushRecording(); // synchronous flush+fence (no batch) → all matvecs done
            }
            nanos = System.nanoTime() - t0;
            if (logitsOut != null) {
                int off = 0;
                for (WarpGpuBuffer o : outs) {
                    float[] part = o.readback();
                    System.arraycopy(part, 0, logitsOut, off, part.length);
                    off += part.length;
                }
            }
        } finally {
            for (WarpGpuBuffer o : outs) {
                o.close();
            }
        }
        return nanos;
    }

    private List<WarpDenseProjection> buildRowBlocks(ByteBuffer embBb, int vocab, int hidden, int blocks) {
        int rowsPer = vocab / blocks;
        List<WarpDenseProjection> parts = new ArrayList<>(blocks);
        for (int b = 0; b < blocks; b++) {
            ByteBuffer dup = embBb.duplicate().order(ByteOrder.LITTLE_ENDIAN);
            dup.position(b * rowsPer * hidden * Float.BYTES);
            dup.limit((b + 1) * rowsPer * hidden * Float.BYTES);
            ByteBuffer blockLe = dup.slice().order(ByteOrder.LITTLE_ENDIAN);
            parts.add(WarpDenseProjection.fromDequantizedWeights(wb, "lmblock" + b, rowsPer, hidden, blockLe));
        }
        return parts;
    }

    private static void assertCloseTop1(String name, float[] base, float[] other) {
        assertEquals(DecoderOnlyMath.argmax(base), DecoderOnlyMath.argmax(other),
                name + " must keep the full-vocab Top-1");
    }

    private static float[] randomVector(Random rng, int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = rng.nextFloat() * 2 - 1;
        }
        return x;
    }

    // ── loaders (same as the gated steady-state test) ──────────────────────────────────────

    private static Gemma3WarpLayerWeights[] loadWarpLayers(SafeTensorsFile f, Gemma3Config c) throws IOException {
        int n = c.numHiddenLayers();
        Gemma3WarpLayerWeights[] out = new Gemma3WarpLayerWeights[n];
        for (int i = 0; i < n; i++) {
            out[i] = new Gemma3WarpLayerWeights(
                    floats(f, Gemma3TensorNameMapper.inputLayerNorm(i)),
                    floats(f, Gemma3TensorNameMapper.qProj(i)), floats(f, Gemma3TensorNameMapper.kProj(i)),
                    floats(f, Gemma3TensorNameMapper.vProj(i)), floats(f, Gemma3TensorNameMapper.oProj(i)),
                    floats(f, Gemma3TensorNameMapper.qNorm(i)), floats(f, Gemma3TensorNameMapper.kNorm(i)),
                    floats(f, Gemma3TensorNameMapper.postAttentionLayerNorm(i)),
                    floats(f, Gemma3TensorNameMapper.preFeedforwardLayerNorm(i)),
                    floats(f, Gemma3TensorNameMapper.gateProj(i)), floats(f, Gemma3TensorNameMapper.upProj(i)),
                    floats(f, Gemma3TensorNameMapper.downProj(i)),
                    floats(f, Gemma3TensorNameMapper.postFeedforwardLayerNorm(i)));
        }
        return out;
    }

    private static float[] floats(SafeTensorsFile f, String name) throws IOException {
        SafeTensorEntry e = f.tensors().get(name);
        if (e == null) {
            throw new IOException("missing tensor: " + name);
        }
        return Gemma3ReferenceWeights.decodeFloats(e);
    }

    private static ByteBuffer decodeEmbeddingDirectFp32(SafeTensorEntry e) {
        long count = 1;
        for (long d : e.shape()) {
            count *= d;
        }
        int n = Math.toIntExact(count);
        ByteBuffer out = ByteBuffer.allocateDirect(n * 4).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer src = e.dataBuffer();
        switch (e.dtype()) {
            case "F32" -> {
                for (int i = 0; i < n; i++) {
                    out.putFloat(src.getFloat());
                }
            }
            case "F16" -> {
                for (int i = 0; i < n; i++) {
                    out.putFloat(Float.float16ToFloat(src.getShort()));
                }
            }
            case "BF16" -> {
                for (int i = 0; i < n; i++) {
                    out.putFloat(Float.intBitsToFloat((src.getShort() & 0xFFFF) << 16));
                }
            }
            default -> throw new IllegalArgumentException("unsupported embedding dtype: " + e.dtype());
        }
        out.flip();
        return out;
    }

    private static Path resolveModelDir() {
        String override = System.getProperty("gemma.testModelDir");
        if (override != null && !override.isBlank()) {
            return dirIfValid(Path.of(override));
        }
        String appData = System.getenv("APPDATA");
        if (appData != null && !appData.isBlank()) {
            Path p = dirIfValid(Path.of(appData, ".directml", "model", "gemma-3-270m-it"));
            if (p != null) {
                return p;
            }
        }
        String home = System.getProperty("user.home");
        return home == null ? null : dirIfValid(Path.of(home, ".directml", "model", "gemma-3-270m-it"));
    }

    private static Path dirIfValid(Path dir) {
        return dir != null && Files.isRegularFile(dir.resolve("config.json"))
                && Files.isRegularFile(dir.resolve("model.safetensors")) ? dir : null;
    }
}
