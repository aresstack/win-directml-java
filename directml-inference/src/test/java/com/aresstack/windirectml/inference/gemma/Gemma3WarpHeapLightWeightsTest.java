package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-13a: heap-light weight load.
 *
 * <ul>
 *   <li>Synthetic: ByteBuffer-projection layer weights produce <b>identical</b> logits to the
 *       {@code float[]} path (same FP32 bytes, just uploaded from off-heap buffers).</li>
 *   <li>Real model: {@code Gemma3RuntimePackage.loadWarpWeightsHeapLight()} yields ByteBuffer-backed
 *       weights, the native run still produces " Paris", and the on-heap footprint after load is logged
 *       (well below the ~1.2 GB float[] reference path measured in GEMMA-WARP-12).</li>
 * </ul>
 *
 * <p>Skipped (assumption-aborted) without a DirectML/D3D12 device.</p>
 */
@EnabledOnOs(OS.WINDOWS)
class Gemma3WarpHeapLightWeightsTest {

    private static final int[] FRANCE_IDS = {2, 818, 5279, 529, 7001, 563};
    private static final int EXPECTED_NEXT = 9079; // " Paris"

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
        return new Gemma3Config("gemma3_text", List.of("Gemma3ForCausalLM"),
                8, 16, 4, 2, 1, 4, 32, 32768, 1e-6, 1_000_000, 10_000, 512, 2, List.of(),
                4, "gelu_pytorch_tanh", 2, 1, 0, true);
    }

    @Test
    void byteBufferProjectionsMatchFloatArrayPath() throws Exception {
        Gemma3Config config = smallConfig();
        Random rng = new Random(1301);
        Gemma3ReferenceWeights ref = syntheticWeights(config, rng);
        int[] ids = {7, 3, 19, 0, 25};

        Gemma3WarpWeights floatW = Gemma3WarpWeights.from(ref);
        Gemma3WarpWeights bbW = toByteBufferWeights(config, ref);
        assertTrue(bbW.hasByteBufferEmbedding(), "embedding should be ByteBuffer-backed");

        try (Gemma3WarpForwardPass fp = new Gemma3WarpForwardPass(wb, floatW);
             Gemma3WarpForwardPass bp = new Gemma3WarpForwardPass(wb, bbW)) {
            float[] lf = fp.logitsForLastToken(ids);
            float[] lb = bp.logitsForLastToken(ids);
            for (int o = 0; o < lf.length; o++) {
                assertEquals(lf[o], lb[o], 1e-6f, "logits[" + o + "] float vs byteBuffer");
            }
            assertEquals(DecoderOnlyMath.argmax(lf), DecoderOnlyMath.argmax(lb), "top-1");
        }
    }

    @EnabledIfSystemProperty(named = "gemma.warp.realModel", matches = "true")
    @Test
    void realModelHeapLightLoadKeepsParisAndLowerHeap() throws Exception {
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");

        Path pkg = Files.createTempFile("gemma3-heaplight-", ".wdmlpack");
        try {
            Gemma3WdmlPackCompiler.compile(dir, pkg, true);
            Gemma3RuntimePackage rp = Gemma3RuntimePackage.open(pkg);

            System.gc();
            long heapBefore = usedHeapMB();
            Gemma3WarpWeights weights = rp.loadWarpWeightsHeapLight();
            System.gc();
            long heapAfter = usedHeapMB();
            System.out.println("[HEAPLIGHT] usedHeapMB before=" + heapBefore + " afterLoad=" + heapAfter
                    + " delta=" + (heapAfter - heapBefore) + " (float[] reference path was ~1199 MB in WARP-12)");

            assertTrue(weights.hasByteBufferEmbedding(), "embedding must be ByteBuffer-backed");
            assertTrue(weights.layers()[0].hasByteBufferProjections(), "layer projections must be ByteBuffer-backed");

            try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, weights)) {
                int top1 = sess.prefillNextToken(FRANCE_IDS);
                System.out.println("[HEAPLIGHT] prefill top-1 = " + top1 + " (expected " + EXPECTED_NEXT + ")");
                assertEquals(EXPECTED_NEXT, top1, "heap-light native prefill top-1 must be \" Paris\"");
            }
        } finally {
            Files.deleteIfExists(pkg);
        }
    }

    // ── helpers ──────────────────────────────────────────────────────

    private static Gemma3WarpWeights toByteBufferWeights(Gemma3Config c, Gemma3ReferenceWeights ref) {
        Gemma3WarpLayerWeights[] layers = new Gemma3WarpLayerWeights[ref.layers.length];
        for (int i = 0; i < layers.length; i++) {
            Gemma3ReferenceWeights.Layer l = ref.layers[i];
            layers[i] = Gemma3WarpLayerWeights.ofByteBufferProjections(
                    l.inputLayerNorm,
                    fp32(l.qProj), fp32(l.kProj), fp32(l.vProj), fp32(l.oProj),
                    l.qNorm, l.kNorm, l.postAttentionLayerNorm, l.preFeedforwardLayerNorm,
                    fp32(l.gateProj), fp32(l.upProj), fp32(l.downProj),
                    l.postFeedforwardLayerNorm);
        }
        return Gemma3WarpWeights.ofByteBufferEmbedding(c, fp32(ref.embedTokens), ref.finalNorm, layers);
    }

    private static ByteBuffer fp32(float[] a) {
        ByteBuffer b = ByteBuffer.allocateDirect(a.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : a) {
            b.putFloat(v);
        }
        b.flip();
        return b;
    }

    private static long usedHeapMB() {
        Runtime r = Runtime.getRuntime();
        return (r.totalMemory() - r.freeMemory()) >> 20;
    }

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
