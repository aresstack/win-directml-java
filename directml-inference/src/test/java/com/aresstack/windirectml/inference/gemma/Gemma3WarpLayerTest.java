package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-8: GPU correctness of a full single Gemma 3 layer ({@link Gemma3WarpLayer}) against the
 * verified CPU reference ({@code Gemma3ReferenceForwardPass.runLayer}).
 *
 * <p>A single layer chains many WARP kernels (RMSNorm, q/k/v proj, QK-norm, RoPE, scores, softmax,
 * value, o_proj, MLP). The kernels compute in {@code float} while the reference accumulates the norm /
 * RoPE / softmax in {@code double}, so the documented single-layer tolerance is looser than the
 * element-wise kernels. Skipped (assumption-aborted) without a DirectML/D3D12 device.</p>
 */
@EnabledOnOs(OS.WINDOWS)
class Gemma3WarpLayerTest {

    // Single layer = a long float chain vs a partly-double reference. Documented, not a fabricated parity.
    private static final float ABS_TOL = 2e-3f;
    private static final float REL_TOL = 2e-3f;

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

    private static Gemma3Config config(int hidden, int intermediate, int heads, int kv, int headDim, int window) {
        return new Gemma3Config("gemma3_text", List.of("Gemma3ForCausalLM"),
                hidden, intermediate, 18, heads, kv, headDim, 32, 32768, 1e-6,
                1_000_000, 10_000, window, 6, List.of(), 256, "gelu_pytorch_tanh", 2, 1, 0, true);
    }

    @Test
    void tinyLocalLayerMatchesReference() throws Exception {
        // local layer 0, small sliding window so the mask actually bites
        assertLayerMatchesReference(config(8, 16, 2, 1, 4, 2), 0, 4, new Random(301));
    }

    @Test
    void tinyFullLayerMatchesReference() throws Exception {
        // full-attention layer 5 (pattern 6)
        assertLayerMatchesReference(config(8, 16, 2, 1, 4, 512), 5, 4, new Random(303));
    }

    @Test
    void gqaLayerMatchesReference() throws Exception {
        // 4 query heads / 2 kv heads
        assertLayerMatchesReference(config(16, 32, 4, 2, 4, 512), 5, 5, new Random(307));
    }

    @Test
    void realHeadDimFullLayerMatchesReference() throws Exception {
        // real Gemma 3 270M layer geometry: hidden=640, 4 heads / 1 kv, head_dim=256, inter=2048
        assertLayerMatchesReference(config(640, 2048, 4, 1, 256, 512), 5, 3, new Random(311));
    }

    @Test
    void realHeadDimLocalLayerMatchesReference() throws Exception {
        // same geometry, local layer with a biting window
        assertLayerMatchesReference(config(640, 2048, 4, 1, 256, 2), 0, 4, new Random(313));
    }

    // ── helpers ──────────────────────────────────────────────────────

    private void assertLayerMatchesReference(Gemma3Config config, int layerIndex, int seqLen, Random rng)
            throws Exception {
        int h = config.hiddenSize();
        Gemma3ReferenceWeights.Layer layer = syntheticLayer(config, rng);

        // reference path
        Gemma3ReferenceWeights.Layer[] layers = new Gemma3ReferenceWeights.Layer[config.numHiddenLayers()];
        layers[layerIndex] = layer;
        Gemma3ReferenceWeights refWeights = new Gemma3ReferenceWeights(config, new float[]{0f}, new float[h], layers);
        Gemma3ReferenceForwardPass ref = new Gemma3ReferenceForwardPass(refWeights);

        float[][] input = new float[seqLen][h];
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < h; i++) {
                input[t][i] = rng.nextFloat() * 2 - 1;
            }
        }
        float[][] expected = deepCopy(input);
        ref.runLayer(expected, layerIndex);

        // WARP path
        float[][] gpu = deepCopy(input);
        try (Gemma3WarpLayer warp = new Gemma3WarpLayer(wb, config, layerIndex,
                Gemma3WarpLayerWeights.from(layer))) {
            warp.forward(gpu);
        }

        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < h; i++) {
                assertClose("layer out [t=" + t + ", i=" + i + "] (h=" + h + ", layer=" + layerIndex + ")",
                        expected[t][i], gpu[t][i]);
            }
        }
    }

    private static Gemma3ReferenceWeights.Layer syntheticLayer(Gemma3Config c, Random rng) {
        int h = c.hiddenSize();
        int d = c.headDim();
        int attnDim = c.attentionDim();
        int kvDim = c.keyValueDim();
        int inter = c.intermediateSize();
        float wRange = 0.05f;   // small projection weights (the normed-input operating regime)
        float nRange = 0.05f;   // zero-centered norm weights: scale is (1 + w)
        return new Gemma3ReferenceWeights.Layer(
                rand(rng, h, nRange),
                rand(rng, attnDim * h, wRange), rand(rng, kvDim * h, wRange), rand(rng, kvDim * h, wRange),
                rand(rng, h * attnDim, wRange),
                rand(rng, d, nRange), rand(rng, d, nRange),
                rand(rng, h, nRange), rand(rng, h, nRange),
                rand(rng, inter * h, wRange), rand(rng, inter * h, wRange), rand(rng, h * inter, wRange),
                rand(rng, h, nRange));
    }

    private static float[] rand(Random rng, int n, float range) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (rng.nextFloat() * 2 - 1) * range;
        }
        return v;
    }

    private static float[][] deepCopy(float[][] a) {
        float[][] c = new float[a.length][];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i].clone();
        }
        return c;
    }

    private static void assertClose(String label, float want, float got) {
        float tol = ABS_TOL + REL_TOL * Math.abs(want);
        float diff = Math.abs(want - got);
        if (diff > tol) {
            fail(String.format("%s mismatch: want=%.6f got=%.6f diff=%.3e (tol=%.3e)",
                    label, want, got, diff, tol));
        }
        assertEquals(want, got, tol, label);
    }
}
