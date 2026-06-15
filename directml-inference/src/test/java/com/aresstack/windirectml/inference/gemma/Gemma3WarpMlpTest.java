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
 * GEMMA-WARP-6: GPU correctness of the Gemma GELU-tanh / GeGLU activation kernels and the composed
 * {@link Gemma3WarpMlp} against the verified CPU reference ({@link Gemma3ReferenceMath} +
 * the GeGLU MLP body of {@code Gemma3ReferenceForwardPass}).
 *
 * <p>Skipped (assumption-aborted, not failed) without a DirectML/D3D12 device, so it stays green on
 * CI; runs on a Windows host with a working adapter. The GPU computes in {@code float}.</p>
 */
@EnabledOnOs(OS.WINDOWS)
class Gemma3WarpMlpTest {

    // Element-wise activations: GPU float vs reference double tanh — tight.
    private static final float ACT_ABS_TOL = 1e-4f;
    private static final float ACT_REL_TOL = 1e-4f;
    // Full MLP: three float matmuls (intermediate=2048) accumulate in a different order than the
    // reference float dot, so a looser, documented tolerance applies.
    private static final float MLP_ABS_TOL = 1e-3f;
    private static final float MLP_REL_TOL = 1e-3f;

    private static final int GEMMA_HIDDEN = 640;        // real Gemma 3 270M hidden_size
    private static final int GEMMA_INTERMEDIATE = 2048; // real Gemma 3 270M intermediate_size

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
    void geluTanhMatchesReferenceIncludingZeroAndSignedValues() throws Exception {
        float[] x = {-4f, -2f, -1f, -0.5f, 0f, 0.5f, 1f, 2f, 4f, 7.3f, -7.3f};
        try (Gemma3WarpGeluTanhKernel kernel = new Gemma3WarpGeluTanhKernel(wb)) {
            float[] gpu = kernel.apply(x);
            for (int i = 0; i < x.length; i++) {
                assertClose("gelu[" + i + "] x=" + x[i], Gemma3ReferenceMath.geluTanh(x[i]), gpu[i],
                        ACT_ABS_TOL, ACT_REL_TOL);
            }
        }
    }

    @Test
    void geluTanhMatchesReferenceOnRandomIntermediateShape() throws Exception {
        Random rng = new Random(31);
        float[] x = randomTensor(rng, GEMMA_INTERMEDIATE, 6f); // wider range to exercise the tails
        try (Gemma3WarpGeluTanhKernel kernel = new Gemma3WarpGeluTanhKernel(wb)) {
            float[] gpu = kernel.apply(x);
            for (int i = 0; i < x.length; i++) {
                assertClose("gelu[" + i + "]", Gemma3ReferenceMath.geluTanh(x[i]), gpu[i],
                        ACT_ABS_TOL, ACT_REL_TOL);
            }
        }
    }

    @Test
    void geGluMatchesReferenceOnRealIntermediate() throws Exception {
        Random rng = new Random(37);
        int inter = GEMMA_INTERMEDIATE;
        float[] gate = randomTensor(rng, inter, 5f);
        float[] up = randomTensor(rng, inter, 5f);
        float[] gateUp = new float[2 * inter];
        System.arraycopy(gate, 0, gateUp, 0, inter);
        System.arraycopy(up, 0, gateUp, inter, inter);

        try (Gemma3WarpGeGluKernel kernel = new Gemma3WarpGeGluKernel(wb)) {
            float[] gpu = kernel.apply(gateUp, inter);
            // reference: gelu_tanh(gate) * up
            float[] expected = gate.clone();
            Gemma3ReferenceMath.geluTanhInPlace(expected);
            Gemma3ReferenceMath.multiplyInPlace(expected, up);
            for (int i = 0; i < inter; i++) {
                assertClose("geglu[" + i + "]", expected[i], gpu[i], ACT_ABS_TOL, ACT_REL_TOL);
            }
        }
    }

    @Test
    void geGluWithZeroGateIsZero() throws Exception {
        int inter = 16;
        float[] gateUp = new float[2 * inter]; // gate=0, up=0
        for (int i = 0; i < inter; i++) {
            gateUp[inter + i] = 1.0f; // up=1; gate stays 0 -> gelu(0)=0 -> out 0
        }
        try (Gemma3WarpGeGluKernel kernel = new Gemma3WarpGeGluKernel(wb)) {
            float[] gpu = kernel.apply(gateUp, inter);
            for (int i = 0; i < inter; i++) {
                assertClose("geglu-zero[" + i + "]", 0.0f, gpu[i], ACT_ABS_TOL, ACT_REL_TOL);
            }
        }
    }

    @Test
    void mlpMatchesReferenceOnSmallShape() throws Exception {
        assertMlpCloseToReference(4, 8, new Random(41), 1f);
    }

    @Test
    void mlpMatchesReferenceOnRealGemmaShape() throws Exception {
        // Realistic regime: a pre-feedforward-RMSNorm input is ~O(1) and Gemma's projection weights
        // are small. Range-1 weights over 640/2048 dims would blow activations far past the model's
        // operating range (and stress float in a way the model never sees), so weights stay small.
        assertMlpCloseToReference(GEMMA_HIDDEN, GEMMA_INTERMEDIATE, new Random(43), 0.05f);
    }

    // ── helpers ──────────────────────────────────────────────────────

    private void assertMlpCloseToReference(int hidden, int intermediate, Random rng, float weightRange)
            throws Exception {
        float[] x = randomTensor(rng, hidden, 1f);
        float[] gateProj = randomTensor(rng, intermediate * hidden, weightRange);
        float[] upProj = randomTensor(rng, intermediate * hidden, weightRange);
        float[] downProj = randomTensor(rng, hidden * intermediate, weightRange);

        float[] expected = referenceMlp(x, hidden, intermediate, gateProj, upProj, downProj);

        try (Gemma3WarpMlp mlp = new Gemma3WarpMlp(wb, hidden, intermediate, gateProj, upProj, downProj)) {
            float[] gpu = mlp.mlp(x);
            assertEquals(hidden, gpu.length, "mlp output width");
            for (int i = 0; i < hidden; i++) {
                assertClose("mlp[" + i + "] (h=" + hidden + ", inter=" + intermediate + ")",
                        expected[i], gpu[i], MLP_ABS_TOL, MLP_REL_TOL);
            }
        }
    }

    /** CPU reference matching the GeGLU MLP body of Gemma3ReferenceForwardPass (float accumulation). */
    private static float[] referenceMlp(float[] x, int hidden, int intermediate,
                                        float[] gateProj, float[] upProj, float[] downProj) {
        float[] gate = matvec(gateProj, x, intermediate, hidden);
        float[] up = matvec(upProj, x, intermediate, hidden);
        Gemma3ReferenceMath.geluTanhInPlace(gate);
        Gemma3ReferenceMath.multiplyInPlace(gate, up);
        return matvec(downProj, gate, hidden, intermediate);
    }

    /** y[o] = sum_i W[o*in + i] * x[i] (float, same as the reference dot). */
    private static float[] matvec(float[] w, float[] x, int out, int in) {
        float[] y = new float[out];
        for (int o = 0; o < out; o++) {
            float sum = 0;
            int base = o * in;
            for (int i = 0; i < in; i++) {
                sum += w[base + i] * x[i];
            }
            y[o] = sum;
        }
        return y;
    }

    private static float[] randomTensor(Random rng, int n, float range) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (rng.nextFloat() * 2 - 1) * range;
        }
        return v;
    }

    private static void assertClose(String label, float want, float got, float absTol, float relTol) {
        float tol = absTol + relTol * Math.abs(want);
        float diff = Math.abs(want - got);
        if (diff > tol) {
            fail(String.format("%s mismatch: want=%.6f got=%.6f diff=%.3e (tol=%.3e)",
                    label, want, got, diff, tol));
        }
        assertEquals(want, got, tol, label);
    }
}
