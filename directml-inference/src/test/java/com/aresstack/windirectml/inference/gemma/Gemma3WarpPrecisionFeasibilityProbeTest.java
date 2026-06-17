package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorEntry;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorsFile;
import com.aresstack.windirectml.inference.warp.WarpDenseProjection;
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
 * GEMMA-WARP-19: reduced-precision feasibility probe for the WARP LM head (and, by extension, the GEMMs).
 * Measure-only; no production quantization. Gated:
 * {@code -Dgemma.warp.realModel=true -Dgemma.warp.quantProbe=true}, explicit WARP adapter.
 *
 * <p>What this measures cheaply with the existing FP32 DML-GEMM: the <b>numeric</b> effect of rounding the
 * LM-head weights to FP16 / BF16 (run through the same GEMM) — Top-1 preservation over the full vocab, logit
 * deviation, and the theoretical weight-size saving. This is the precondition for any future quantization.
 * It does <b>not</b> measure FP16 GEMM <i>speed</i>: the DirectML GEMM operator is wired FP32-only, so a
 * real FP16 path would need new operator wiring — that, plus the WARP-CPU-is-compute-bound finding and the
 * already-measured INT4-slower-on-WARP result (GEMMA-WARP-14a), is the basis for the decision documented in
 * {@code docs/gemma3-warp-performance-ceiling.md} §11.</p>
 */
@EnabledOnOs(OS.WINDOWS)
@EnabledIfSystemProperty(named = "gemma.warp.realModel", matches = "true")
class Gemma3WarpPrecisionFeasibilityProbeTest {

    private static final int[] FRANCE_IDS = {2, 818, 5279, 529, 7001, 563};
    private static final int EXPECTED_NEXT = 9079; // " Paris"

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

    private record Variant(String name, String operatorPath, double weightMb, int top1,
                           double maxAbsDev, double meanAbsDev) {
    }

    @Test
    void precisionFeasibilityProbe() throws Exception {
        assumeTrue(Boolean.getBoolean("gemma.warp.quantProbe"), "opt-in via -Dgemma.warp.quantProbe=true");
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");

        Gemma3Config config = new Gemma3ConfigReader().read(dir.resolve("config.json"));
        int vocab = config.vocabSize();
        int hidden = config.hiddenSize();
        SafeTensorsFile file = SafeTensorsReader.read(dir.resolve("model.safetensors"));
        ByteBuffer embBb = decodeEmbeddingDirectFp32(file.tensors().get(Gemma3TensorNameMapper.EMBED_TOKENS));

        // ── Pflicht-Smoke: the FP32 product path gives " Paris" ──
        {
            float[] finalNorm = Gemma3ReferenceWeights.decodeFloats(file.tensors().get(Gemma3TensorNameMapper.FINAL_NORM));
            Gemma3WarpLayerWeights[] layers = loadWarpLayers(file, config);
            Gemma3WarpWeights weights = Gemma3WarpWeights.ofByteBufferEmbedding(config, embBb, finalNorm, layers);
            try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, weights)) {
                assertEquals(EXPECTED_NEXT, DecoderOnlyMath.argmax(sess.prefillResidentBatched(FRANCE_IDS)),
                        "FP32 WARP product path must give \" Paris\"");
            }
        }

        // ── Numeric precision probe on the LM head (run rounded weights through the FP32 GEMM) ──
        float[] x = randomVector(new Random(19), hidden);
        double fp32Mb = (double) vocab * hidden * Float.BYTES / (1024 * 1024);
        double half16Mb = (double) vocab * hidden * 2 / (1024 * 1024);

        float[] fp32Logits = lmHeadLogits(embBb, vocab, hidden, x); // baseline
        int baseTop1 = DecoderOnlyMath.argmax(fp32Logits);

        List<Variant> variants = new ArrayList<>();
        variants.add(new Variant("fp32 (baseline, DML GEMM)", "DML_OPERATOR_GEMM FLOAT32", fp32Mb, baseTop1, 0, 0));

        float[] fp16Logits = lmHeadLogits(roundWeights(embBb, Precision.FP16), vocab, hidden, x);
        variants.add(deviation("fp16 weights (DML GEMM, fp32 accumulate)",
                "DML_OPERATOR_GEMM FLOAT32 (fp16-rounded weights)", half16Mb, fp32Logits, fp16Logits));

        float[] bf16Logits = lmHeadLogits(roundWeights(embBb, Precision.BF16), vocab, hidden, x);
        variants.add(deviation("bf16 weights (DML GEMM, fp32 accumulate)",
                "DML_OPERATOR_GEMM FLOAT32 (bf16-rounded weights)", half16Mb, fp32Logits, bf16Logits));

        // ── Report ──
        System.out.println("===== [GEMMA-WARP-19] WARP reduced-precision feasibility (LM head) =====");
        System.out.println("note: speed not measured here — all variants run the FP32 DML GEMM; a real FP16/INT");
        System.out.println("      GEMM needs a new operator path, and WARP is compute-bound (see §11 decision).");
        System.out.printf(Locale.ROOT, "  %-40s %-44s %9s %9s %12s %12s%n",
                "variant", "operator path", "weight MB", "top1", "maxAbsDev", "meanAbsDev");
        for (Variant v : variants) {
            System.out.printf(Locale.ROOT, "  %-40s %-44s %9.0f %9d %12.4f %12.6f%n",
                    v.name(), v.operatorPath(), v.weightMb(), v.top1(), v.maxAbsDev(), v.meanAbsDev());
        }
        // FP16 must preserve the full-vocab Top-1 (it is far more than precise enough for logits).
        assertEquals(baseTop1, DecoderOnlyMath.argmax(fp16Logits), "fp16-rounded LM head must keep Top-1");
    }

    private static Variant deviation(String name, String op, double mb, float[] base, float[] other) {
        double max = 0;
        double sum = 0;
        for (int i = 0; i < base.length; i++) {
            double d = Math.abs(base[i] - other[i]);
            max = Math.max(max, d);
            sum += d;
        }
        return new Variant(name, op, mb, DecoderOnlyMath.argmax(other), max, sum / base.length);
    }

    /** Full-vocab logits for input {@code x} via the existing FP32 DML GEMM over the given weight bytes. */
    private float[] lmHeadLogits(ByteBuffer weightsLe, int vocab, int hidden, float[] x) throws Exception {
        try (WarpDenseProjection proj = WarpDenseProjection.fromDequantizedWeights(
                wb, "lmhead.probe", vocab, hidden, weightsLe.duplicate().order(ByteOrder.LITTLE_ENDIAN));
             WarpGpuBuffer xBuf = WarpGpuBuffer.upload(wb, x)) {
            WarpGpuBuffer out = proj.forwardResident(new com.aresstack.windirectml.windows.WarpExecutionContext(wb), xBuf);
            try {
                return out.readback();
            } finally {
                out.close();
            }
        }
    }

    private enum Precision { FP16, BF16 }

    /** A new little-endian FP32 buffer with every weight rounded to FP16 or BF16 then back to FP32. */
    private static ByteBuffer roundWeights(ByteBuffer fp32Le, Precision p) {
        int n = fp32Le.remaining() / Float.BYTES;
        ByteBuffer src = fp32Le.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer out = ByteBuffer.allocateDirect(n * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            float v = src.getFloat();
            out.putFloat(p == Precision.FP16 ? roundFp16(v) : roundBf16(v));
        }
        out.flip();
        return out;
    }

    private static float roundFp16(float v) {
        return Float.float16ToFloat(Float.floatToFloat16(v));
    }

    /** BF16 = top 16 bits of the float, round-to-nearest-even. */
    private static float roundBf16(float v) {
        int bits = Float.floatToRawIntBits(v);
        int rounding = ((bits >>> 16) & 1) + 0x7FFF;
        return Float.intBitsToFloat((bits + rounding) & 0xFFFF0000);
    }

    private static float[] randomVector(Random rng, int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = rng.nextFloat() * 2 - 1;
        }
        return x;
    }

    // ── loaders (same as the gated steady-state / probe tests) ──────────────────────────────

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
