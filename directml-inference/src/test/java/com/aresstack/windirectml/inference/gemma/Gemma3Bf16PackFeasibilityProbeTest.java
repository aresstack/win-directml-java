package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorEntry;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorsFile;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-BF16-PACK-1: feasibility/accounting probe for lossless 16-bit weight storage. Measure-only — no
 * format change, no product-path change. Gated: {@code -Dgemma.warp.realModel=true -Dgemma.warp.bf16Probe=true}.
 *
 * <p>It confirms which Gemma weights are BF16 in the SafeTensors payload, accounts for the FP32 that the
 * runtime currently materialises (the heap-light loader widens BF16→FP32 into direct ByteBuffers, retained
 * for the embedding lookup + uploaded FP32 to the device for the fp32 GEMM), quantifies the host memory a
 * BF16-retained path would save, and verifies a BF16→FP32 inflate is byte-identical to the current FP32
 * decode ({@link Gemma3WeightBufferView#decodeFp32LittleEndian}). The FP32 product path still gives
 * " Paris" (token 9079); the decision is documented in {@code docs/gemma3-warp-performance-ceiling.md} §13.</p>
 */
@EnabledOnOs(OS.WINDOWS)
@EnabledIfSystemProperty(named = "gemma.warp.realModel", matches = "true")
class Gemma3Bf16PackFeasibilityProbeTest {

    private static final int[] FRANCE_IDS = {2, 818, 5279, 529, 7001, 563};
    private static final int EXPECTED_NEXT = 9079; // " Paris"

    private static WindowsBindings wb;

    @BeforeAll
    static void initGpu() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML device");
        wb = new WindowsBindings();
        wb.init("directml", WindowsBindings.AdapterMode.WARP);
    }

    @AfterAll
    static void closeGpu() {
        if (wb != null) {
            wb.close();
        }
    }

    @Test
    void bf16PackFeasibility() throws Exception {
        assumeTrue(Boolean.getBoolean("gemma.warp.bf16Probe"), "opt-in via -Dgemma.warp.bf16Probe=true");
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");

        Gemma3Config config = new Gemma3ConfigReader().read(dir.resolve("config.json"));
        SafeTensorsFile file = SafeTensorsReader.read(dir.resolve("model.safetensors"));

        // ── Account for the BF16 source bytes vs the FP32 the runtime widens to, per group ──
        long embBf16 = 0, embFp32 = 0;     // tied embedding / LM head (retained FP32 for lookup + GPU)
        long projBf16 = 0, projFp32 = 0;   // q/k/v/o + gate/up/down projections (uploaded FP32 to device)
        long normBytes = 0;                // norm vectors stay float[] (small)
        boolean allBf16 = true;

        SafeTensorEntry emb = entry(file, Gemma3TensorNameMapper.EMBED_TOKENS);
        allBf16 &= "BF16".equals(emb.dtype());
        embBf16 += srcBytes(emb);
        embFp32 += fp32Bytes(emb);

        int n = config.numHiddenLayers();
        for (int i = 0; i < n; i++) {
            for (String name : new String[]{
                    Gemma3TensorNameMapper.qProj(i), Gemma3TensorNameMapper.kProj(i),
                    Gemma3TensorNameMapper.vProj(i), Gemma3TensorNameMapper.oProj(i),
                    Gemma3TensorNameMapper.gateProj(i), Gemma3TensorNameMapper.upProj(i),
                    Gemma3TensorNameMapper.downProj(i)}) {
                SafeTensorEntry e = entry(file, name);
                allBf16 &= "BF16".equals(e.dtype());
                projBf16 += srcBytes(e);
                projFp32 += fp32Bytes(e);
            }
            for (String name : new String[]{
                    Gemma3TensorNameMapper.inputLayerNorm(i), Gemma3TensorNameMapper.qNorm(i),
                    Gemma3TensorNameMapper.kNorm(i), Gemma3TensorNameMapper.postAttentionLayerNorm(i),
                    Gemma3TensorNameMapper.preFeedforwardLayerNorm(i),
                    Gemma3TensorNameMapper.postFeedforwardLayerNorm(i)}) {
                normBytes += fp32Bytes(entry(file, name));
            }
        }
        normBytes += fp32Bytes(entry(file, Gemma3TensorNameMapper.FINAL_NORM));

        // ── Lossless check: BF16→FP32 inflate == the current FP32 decode (first rows of the embedding) ──
        ByteBuffer fp32 = Gemma3WeightBufferView.decodeFp32LittleEndian(emb);
        ByteBuffer bf16 = emb.dataBuffer(); // raw BF16 (2 bytes/elem)
        int sample = Math.min(config.hiddenSize() * 64, fp32.limit() / Float.BYTES);
        for (int i = 0; i < sample; i++) {
            float widened = fp32.getFloat(i * Float.BYTES);
            float inflated = Float.intBitsToFloat((bf16.getShort(bf16.position() + i * 2) & 0xFFFF) << 16);
            assertEquals(widened, inflated, 0f, "BF16->FP32 inflate must equal the FP32 decode at " + i);
        }

        // ── Pflicht-Smoke: the FP32 product path gives " Paris" ──
        float[] finalNorm = Gemma3ReferenceWeights.decodeFloats(entry(file, Gemma3TensorNameMapper.FINAL_NORM));
        Gemma3WarpLayerWeights[] layers = loadWarpLayers(file, config);
        ByteBuffer embFp32Bb = Gemma3WeightBufferView.decodeFp32LittleEndian(emb);
        Gemma3WarpWeights weights = Gemma3WarpWeights.ofByteBufferEmbedding(config, embFp32Bb, finalNorm, layers);
        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, weights)) {
            assertEquals(EXPECTED_NEXT, DecoderOnlyMath.argmax(sess.prefillResidentBatched(FRANCE_IDS)),
                    "FP32 product path must give \" Paris\"");
        }

        // ── Report ──
        double mb = 1024.0 * 1024.0;
        System.out.println("===== [GEMMA-BF16-PACK-1] Gemma weight precision / packaging accounting =====");
        System.out.printf(Locale.ROOT, "all matmul weights BF16 in safetensors: %b; .wdmlpack payload = verbatim "
                + "safetensors → already BF16 on disk (no disk saving available)%n", allBf16);
        System.out.printf(Locale.ROOT, "  %-34s %12s %12s %12s%n", "group", "BF16 MB", "FP32 MB", "host save MB");
        row("tied embedding / LM head", embBf16, embFp32, mb);
        row("layer projections (q/k/v/o,gate/up,down)", projBf16, projFp32, mb);
        System.out.printf(Locale.ROOT, "  %-34s %12s %12.1f %12s%n", "norm vectors (float[], small)", "-",
                normBytes / mb, "-");
        System.out.printf(Locale.ROOT, "  %-34s %12.1f %12.1f %12.1f%n", "TOTAL matmul weights",
                (embBf16 + projBf16) / mb, (embFp32 + projFp32) / mb, (embFp32 + projFp32 - embBf16 - projBf16) / mb);
        System.out.println("note: device/GPU buffers stay FP32 (the DML GEMM is fp32) — the saving above is the");
        System.out.println("      host/system-RAM decoded weight buffers (the retained FP32 embedding lookup is the");
        System.out.println("      single biggest item). GPU-side BF16 would need an fp16 GEMM operator (out of scope,");
        System.out.println("      no speed benefit per §11). The disk package is already BF16 (verbatim safetensors).");
    }

    private static void row(String label, long bf16, long fp32, double mb) {
        System.out.printf(Locale.ROOT, "  %-34s %12.1f %12.1f %12.1f%n",
                label, bf16 / mb, fp32 / mb, (fp32 - bf16) / mb);
    }

    private static long srcBytes(SafeTensorEntry e) {
        return e.dataBuffer().remaining(); // BF16 = 2 bytes/element
    }

    private static long fp32Bytes(SafeTensorEntry e) {
        long count = 1;
        for (long d : e.shape()) {
            count *= d;
        }
        return count * Float.BYTES;
    }

    private static SafeTensorEntry entry(SafeTensorsFile f, String name) throws IOException {
        SafeTensorEntry e = f.tensors().get(name);
        if (e == null) {
            throw new IOException("missing tensor: " + name);
        }
        return e;
    }

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
        return Gemma3ReferenceWeights.decodeFloats(entry(f, name));
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
