package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorEntry;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorsFile;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-9: full WARP prefill (embedding ×sqrt(hidden) → layers → final RMSNorm → tied LM head)
 * against the verified CPU reference.
 *
 * <ul>
 *   <li><b>Synthetic, all layers + LM head</b>: proves the full mechanism exactly (embedding scale,
 *       layer loop, final norm, tied LM head, top-1), and that the heap-light ByteBuffer embedding/LM
 *       head is numerically identical to the {@code float[]} path.</li>
 *   <li><b>Real model, partial</b>: runs the first {@code N} real Gemma 3 270M layers (BF16-decoded)
 *       through the WARP stack and checks per-element parity vs the reference — real-weight numerics.</li>
 *   <li><b>Real model, full</b>: the full 18-layer prefill + tied LM head; top-1 must be " Paris"
 *       (token 9079), matching transformers/the reference.</li>
 * </ul>
 *
 * <p>Skipped (assumption-aborted) without a DirectML/D3D12 device. The GPU computes in {@code float}.</p>
 */
@EnabledOnOs(OS.WINDOWS)
class Gemma3WarpPrefillTest {

    // Prefill accumulates over layers vs a partly-double reference — documented, looser than a single layer.
    private static final float ABS_TOL = 3e-3f;
    private static final float REL_TOL = 3e-3f;

    private static final int[] FRANCE_IDS = {2, 818, 5279, 529, 7001, 563};
    private static final int EXPECTED_NEXT = 9079; // " Paris" per transformers/reference
    /** Real layers checked element-wise in the partial test (the full test covers all 18 + top-1). */
    private static final int REAL_PARTIAL_LAYERS = 4;

    private static WindowsBindings wb;

    @BeforeAll
    static void initGpu() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML device");
        wb = new WindowsBindings();
        wb.init("directml");
    }

    @AfterAll
    static void closeGpu() {
        if (wb != null) {
            wb.close();
        }
    }

    private static Gemma3Config smallConfig() {
        // vocab=32, hidden=8, 4 layers, pattern 2 -> layers 1,3 full / 0,2 local; head_dim=4, GQA 2/1.
        return new Gemma3Config("gemma3_text", List.of("Gemma3ForCausalLM"),
                8, 16, 4, 2, 1, 4, 32, 32768, 1e-6, 1_000_000, 10_000, 2, 2, List.of(),
                4, "gelu_pytorch_tanh", 2, 1, 0, true);
    }

    @Test
    void syntheticPrefillMatchesReference() throws Exception {
        Gemma3Config config = smallConfig();
        Random rng = new Random(901);
        Gemma3ReferenceWeights ref = syntheticWeights(config, rng);
        Gemma3ReferenceForwardPass oracle = new Gemma3ReferenceForwardPass(ref);
        int[] ids = {7, 3, 19, 0, 25};

        float[] expectedHidden = oracle.finalHiddenForLastToken(ids);
        float[] expectedLogits = oracle.logitsForLastToken(ids);
        int expectedTop1 = DecoderOnlyMath.argmax(expectedLogits);

        try (Gemma3WarpForwardPass warp = new Gemma3WarpForwardPass(wb, Gemma3WarpWeights.from(ref))) {
            float[] gpuHidden = warp.finalHiddenForLastToken(ids);     // 9a
            for (int i = 0; i < expectedHidden.length; i++) {
                assertClose("finalHidden[" + i + "]", expectedHidden[i], gpuHidden[i]);
            }
            float[] gpuLogits = warp.logitsForLastToken(ids);          // 9b
            for (int o = 0; o < expectedLogits.length; o++) {
                assertClose("logits[" + o + "]", expectedLogits[o], gpuLogits[o]);
            }
            assertEquals(expectedTop1, DecoderOnlyMath.argmax(gpuLogits), "top-1 must match the reference");
        }
    }

    @Test
    void heapLightByteBufferEmbeddingMatchesFloatPath() throws Exception {
        // The heap-light ByteBuffer LM head/embedding must be numerically identical to the float[] path
        // (same FP32 bytes uploaded once).
        Gemma3Config config = smallConfig();
        Random rng = new Random(907);
        Gemma3ReferenceWeights ref = syntheticWeights(config, rng);
        int[] ids = {1, 14, 30, 8};

        Gemma3WarpWeights floatW = Gemma3WarpWeights.from(ref);
        ByteBuffer embBb = ByteBuffer.allocateDirect(ref.embedTokens.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : ref.embedTokens) {
            embBb.putFloat(v);
        }
        embBb.flip();
        Gemma3WarpWeights bbW = Gemma3WarpWeights.ofByteBufferEmbedding(
                config, embBb, ref.finalNorm, floatW.layers());

        try (Gemma3WarpForwardPass fp = new Gemma3WarpForwardPass(wb, floatW);
             Gemma3WarpForwardPass bp = new Gemma3WarpForwardPass(wb, bbW)) {
            float[] lf = fp.logitsForLastToken(ids);
            float[] lb = bp.logitsForLastToken(ids);
            for (int o = 0; o < lf.length; o++) {
                assertEquals(lf[o], lb[o], 1e-6f, "byteBuffer vs float logits[" + o + "]");
            }
        }
    }

    @Test
    void realModelPartialPrefillMatchesReference() throws Exception {
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");
        Gemma3Config config = new Gemma3ConfigReader().read(dir.resolve("config.json"));
        assumeTrue(config.vocabSize() == 262144, "expected the real 270M-it config");
        SafeTensorsFile file = SafeTensorsReader.read(dir.resolve("model.safetensors"));

        int h = config.hiddenSize();
        float scale = (float) config.embeddingScale();
        ByteBuffer embBb = decodeEmbeddingDirectFp32(file.tensors().get(Gemma3TensorNameMapper.EMBED_TOKENS));
        float[] finalNorm = Gemma3ReferenceWeights.decodeFloats(file.tensors().get(Gemma3TensorNameMapper.FINAL_NORM));
        Gemma3ReferenceWeights.Layer[] refLayers = loadRefLayers(file, config);

        // Reference: embed (from the same buffer) then the first N real layers.
        Gemma3ReferenceWeights refWeights = new Gemma3ReferenceWeights(config, new float[]{0f}, finalNorm, refLayers);
        Gemma3ReferenceForwardPass ref = new Gemma3ReferenceForwardPass(refWeights);
        float[][] expected = Gemma3WarpEmbedding.lookupScaled(embBb, FRANCE_IDS, h, scale);
        for (int l = 0; l < REAL_PARTIAL_LAYERS; l++) {
            ref.runLayer(expected, l);
        }

        // WARP: the same first N real layers on the shared kernel bundle.
        float[][] gpu = Gemma3WarpEmbedding.lookupScaled(embBb, FRANCE_IDS, h, scale);
        List<Gemma3WarpLayer> built = new ArrayList<>();
        try (Gemma3WarpKernels kernels = new Gemma3WarpKernels(wb)) {
            for (int l = 0; l < REAL_PARTIAL_LAYERS; l++) {
                built.add(new Gemma3WarpLayer(wb, config, l, Gemma3WarpLayerWeights.from(refLayers[l]), kernels));
            }
            for (Gemma3WarpLayer layer : built) {
                layer.forward(gpu);
            }
        } finally {
            for (Gemma3WarpLayer layer : built) {
                layer.close();
            }
        }

        int s = FRANCE_IDS.length;
        for (int i = 0; i < h; i++) {
            assertClose("real partial hidden[t=" + (s - 1) + ", i=" + i + "]", expected[s - 1][i], gpu[s - 1][i]);
        }
    }

    @Test
    void realModelFullPrefillTop1IsParis() throws Exception {
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");
        Gemma3Config config = new Gemma3ConfigReader().read(dir.resolve("config.json"));
        SafeTensorsFile file = SafeTensorsReader.read(dir.resolve("model.safetensors"));
        ByteBuffer embBb = decodeEmbeddingDirectFp32(file.tensors().get(Gemma3TensorNameMapper.EMBED_TOKENS));
        float[] finalNorm = Gemma3ReferenceWeights.decodeFloats(file.tensors().get(Gemma3TensorNameMapper.FINAL_NORM));
        Gemma3WarpLayerWeights[] layers = loadWarpLayers(file, config);
        Gemma3WarpWeights weights = Gemma3WarpWeights.ofByteBufferEmbedding(config, embBb, finalNorm, layers);
        try (Gemma3WarpForwardPass warp = new Gemma3WarpForwardPass(wb, weights)) {
            assertEquals(EXPECTED_NEXT, warp.nextToken(FRANCE_IDS), "WARP full prefill top-1 must be \" Paris\"");
        }
    }

    // ── real-model loading helpers (heap-light embedding) ─────────────

    private static Gemma3ReferenceWeights.Layer[] loadRefLayers(SafeTensorsFile f, Gemma3Config c) throws IOException {
        int n = c.numHiddenLayers();
        Gemma3ReferenceWeights.Layer[] out = new Gemma3ReferenceWeights.Layer[n];
        for (int i = 0; i < n; i++) {
            out[i] = new Gemma3ReferenceWeights.Layer(
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

    private static Gemma3WarpLayerWeights[] loadWarpLayers(SafeTensorsFile f, Gemma3Config c) throws IOException {
        Gemma3ReferenceWeights.Layer[] ref = loadRefLayers(f, c);
        Gemma3WarpLayerWeights[] out = new Gemma3WarpLayerWeights[ref.length];
        for (int i = 0; i < ref.length; i++) {
            out[i] = Gemma3WarpLayerWeights.from(ref[i]);
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

    // ── synthetic weights ────────────────────────────────────────────

    private static Gemma3ReferenceWeights syntheticWeights(Gemma3Config c, Random rng) {
        int h = c.hiddenSize();
        int vocab = c.vocabSize();
        float[] embed = rand(rng, vocab * h, 0.1f);
        float[] finalNorm = rand(rng, h, 0.05f);
        Gemma3ReferenceWeights.Layer[] layers = new Gemma3ReferenceWeights.Layer[c.numHiddenLayers()];
        int d = c.headDim();
        int attnDim = c.attentionDim();
        int kvDim = c.keyValueDim();
        int inter = c.intermediateSize();
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Gemma3ReferenceWeights.Layer(
                    rand(rng, h, 0.05f),
                    rand(rng, attnDim * h, 0.05f), rand(rng, kvDim * h, 0.05f), rand(rng, kvDim * h, 0.05f),
                    rand(rng, h * attnDim, 0.05f),
                    rand(rng, d, 0.05f), rand(rng, d, 0.05f),
                    rand(rng, h, 0.05f), rand(rng, h, 0.05f),
                    rand(rng, inter * h, 0.05f), rand(rng, inter * h, 0.05f), rand(rng, h * inter, 0.05f),
                    rand(rng, h, 0.05f));
        }
        return new Gemma3ReferenceWeights(c, embed, finalNorm, layers);
    }

    private static float[] rand(Random rng, int n, float range) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (rng.nextFloat() * 2 - 1) * range;
        }
        return v;
    }

    private static void assertClose(String label, float want, float got) {
        float tol = ABS_TOL + REL_TOL * Math.abs(want);
        if (Math.abs(want - got) > tol) {
            fail(String.format("%s mismatch: want=%.6f got=%.6f diff=%.3e (tol=%.3e)",
                    label, want, got, Math.abs(want - got), tol));
        }
        assertEquals(want, got, tol, label);
    }
}
