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
 * GEMMA-WARP-15: the fused zero-centered RMSNorm + residual-add kernel must be byte-equivalent (within the
 * float-vs-double reduction tolerance) to running {@link Gemma3WarpRmsNormKernel} then adding the residual
 * — the two-kernel path it replaces in the resident decode/prefill layer. This is what keeps the Paris
 * smoke (" Paris", token 9079) green after the fusion.
 *
 * <p>Skipped (assumption-aborted) when no DirectML/D3D12 device is present.</p>
 */
@EnabledOnOs(OS.WINDOWS)
class Gemma3WarpFusedNormAddKernelTest {

    private static final float ABS_TOL = 1e-4f;
    private static final float REL_TOL = 1e-4f;
    private static final int GEMMA_HIDDEN = 640;

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
    void fusedNormAddMatchesNormThenAddOnRealHidden() throws Exception {
        assertFusedMatchesReference(new Random(101), GEMMA_HIDDEN, 1e-6f);
    }

    @Test
    void fusedNormAddMatchesNormThenAddOnSmallShape() throws Exception {
        assertFusedMatchesReference(new Random(103), 8, 1e-6f);
    }

    @Test
    void fusedNormAddHandlesMultipleEpsValues() throws Exception {
        for (float eps : new float[]{1e-6f, 1e-5f, 1e-3f}) {
            assertFusedMatchesReference(new Random(107), GEMMA_HIDDEN, eps);
        }
    }

    private void assertFusedMatchesReference(Random rng, int dim, float eps) throws Exception {
        float[] x = randomTensor(rng, dim);
        float[] weight = randomTensor(rng, dim);
        float[] residual = randomTensor(rng, dim);

        // Reference: same operations the two-kernel path runs (rmsNorm then residual + normed).
        float[] expected = x.clone();
        Gemma3ReferenceMath.rmsNormZeroCentered(expected, weight, eps);
        for (int i = 0; i < dim; i++) {
            expected[i] = residual[i] + expected[i];
        }

        try (Gemma3WarpFusedNormAddKernel kernel = new Gemma3WarpFusedNormAddKernel(wb)) {
            float[] gpu = kernel.normAdd(x, weight, residual, eps);
            for (int i = 0; i < dim; i++) {
                assertClose("fusedNormAdd[" + i + "] (eps=" + eps + ", dim=" + dim + ")", expected[i], gpu[i]);
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
