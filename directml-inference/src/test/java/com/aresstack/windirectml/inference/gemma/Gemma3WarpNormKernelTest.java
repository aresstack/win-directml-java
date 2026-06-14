package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-5a: GPU correctness of the Gemma zero-centered RMSNorm and QK-Norm WARP kernels against
 * the already-verified CPU reference ({@link Gemma3ReferenceMath#rmsNormZeroCentered}).
 *
 * <p>Skipped (assumption-aborted, not failed) when no DirectML/D3D12 device is present, so it stays
 * green on CI; it runs on a Windows host with a working adapter. The GPU computes in {@code float},
 * the reference accumulates the sum of squares in {@code double}, so a small tolerance applies.</p>
 */
@EnabledOnOs(OS.WINDOWS)
class Gemma3WarpNormKernelTest {

    /** GPU float vs reference double sum-of-squares — absolute + relative slack. */
    private static final float ABS_TOL = 1e-4f;
    private static final float REL_TOL = 1e-4f;

    private static final int GEMMA_HEAD_DIM = 256; // real Gemma 3 270M head_dim (decoupled from hidden/heads)
    private static final int GEMMA_HIDDEN = 640;   // real Gemma 3 270M hidden_size

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

    @Test
    void rmsNormMatchesReferenceOnSmallShape() throws Exception {
        Random rng = new Random(7);
        float[] x = randomTensor(rng, 8);
        float[] weight = randomTensor(rng, 8);
        assertRmsNormCloseToReference(x, weight, 1e-6f);
    }

    @Test
    void rmsNormMatchesReferenceOnRealHidden() throws Exception {
        Random rng = new Random(11);
        float[] x = randomTensor(rng, GEMMA_HIDDEN);
        float[] weight = randomTensor(rng, GEMMA_HIDDEN);
        assertRmsNormCloseToReference(x, weight, 1e-6f);
    }

    @Test
    void rmsNormHandlesMultipleEpsValues() throws Exception {
        Random rng = new Random(13);
        float[] x = randomTensor(rng, GEMMA_HIDDEN);
        float[] weight = randomTensor(rng, GEMMA_HIDDEN);
        for (float eps : new float[]{1e-6f, 1e-5f, 1e-3f}) {
            assertRmsNormCloseToReference(x, weight, eps);
        }
    }

    @Test
    void rmsNormWithZeroWeightIsPureRmsScale() throws Exception {
        // Zero-centered: (1 + 0) == identity scale, so y_i == x_i * rsqrt(mean(x^2)+eps).
        Random rng = new Random(17);
        float[] x = randomTensor(rng, GEMMA_HIDDEN);
        float[] weight = new float[GEMMA_HIDDEN]; // all zero
        float eps = 1e-6f;

        try (Gemma3WarpRmsNormKernel kernel = new Gemma3WarpRmsNormKernel(wb)) {
            float[] gpu = kernel.normalize(x, weight, eps);

            double sumSq = 0;
            for (float v : x) {
                sumSq += (double) v * v;
            }
            float rmsInv = (float) (1.0 / Math.sqrt(sumSq / x.length + eps));
            for (int i = 0; i < x.length; i++) {
                assertClose("zero-weight[" + i + "]", x[i] * rmsInv, gpu[i]);
            }
        }
    }

    @Test
    void qkNormMatchesReferencePerHeadOnRealGemmaShape() throws Exception {
        // Real Gemma 3 270M: 4 query heads, head_dim=256, shared q_norm weight of length head_dim.
        assertQkNormCloseToReference(4, GEMMA_HEAD_DIM, new Random(19), 1e-6f);
    }

    @Test
    void qkNormMatchesReferenceOnSmallShape() throws Exception {
        assertQkNormCloseToReference(3, 8, new Random(23), 1e-6f);
    }

    @Test
    void qkNormSingleKvHeadMatchesReference() throws Exception {
        // Gemma 3 270M GQA: a single kv head (k_norm) — exercises numHeads=1.
        assertQkNormCloseToReference(1, GEMMA_HEAD_DIM, new Random(29), 1e-6f);
    }

    // ── helpers ──────────────────────────────────────────────────────

    private void assertRmsNormCloseToReference(float[] x, float[] weight, float eps) throws Exception {
        try (Gemma3WarpRmsNormKernel kernel = new Gemma3WarpRmsNormKernel(wb)) {
            float[] gpu = kernel.normalize(x, weight, eps);
            float[] expected = x.clone();
            Gemma3ReferenceMath.rmsNormZeroCentered(expected, weight, eps);
            for (int i = 0; i < expected.length; i++) {
                assertClose("rmsNorm[" + i + "] (eps=" + eps + ", dim=" + x.length + ")",
                        expected[i], gpu[i]);
            }
        }
    }

    private void assertQkNormCloseToReference(int numHeads, int headDim, Random rng, float eps)
            throws Exception {
        float[] heads = randomTensor(rng, numHeads * headDim);
        float[] weight = randomTensor(rng, headDim);
        try (Gemma3WarpQkNormKernel kernel = new Gemma3WarpQkNormKernel(wb)) {
            float[] gpu = kernel.normalizeHeads(heads, numHeads, headDim, weight, eps);
            float[] expected = heads.clone();
            for (int h = 0; h < numHeads; h++) {
                Gemma3ReferenceMath.rmsNormZeroCentered(expected, h * headDim, headDim, weight, eps);
            }
            for (int i = 0; i < expected.length; i++) {
                assertClose("qkNorm[" + i + "] (heads=" + numHeads + ", headDim=" + headDim + ")",
                        expected[i], gpu[i]);
            }
        }
    }

    private static float[] randomTensor(Random rng, int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = rng.nextFloat() * 2 - 1;
        }
        return x;
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
