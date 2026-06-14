package com.aresstack.windirectml.inference.gemma;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** GEMMA-WARP-4: device-free parity tests for the Gemma-specific math primitives. */
class Gemma3ReferenceMathTest {

    @Test
    void zeroCenteredRmsNormScalesByOnePlusWeight() {
        // weight all zero -> output is the pure normalized vector (1+0=1), unlike standard RMSNorm.
        float[] v = {1f, 2f, 3f, 4f};
        float[] w = {0f, 0f, 0f, 0f};
        Gemma3ReferenceMath.rmsNormZeroCentered(v, w, 1e-6f);
        double meanSq = (1 + 4 + 9 + 16) / 4.0;
        double rms = 1.0 / Math.sqrt(meanSq + 1e-6);
        assertEquals(1f * rms, v[0], 1e-5);
        assertEquals(4f * rms, v[3], 1e-5);
        // RMS of the normalized vector is ~1.
        double sq = 0;
        for (float f : v) sq += f * f;
        assertEquals(1.0, Math.sqrt(sq / v.length), 1e-4);
    }

    @Test
    void zeroCenteredRmsNormAddsOneToWeight() {
        // With weight = 1 everywhere, Gemma multiplies by (1+1)=2 vs standard which would multiply by 1.
        float[] v = {1f, 2f, 3f, 4f};
        float[] vCopy = v.clone();
        float[] wOne = {1f, 1f, 1f, 1f};
        float[] wZero = {0f, 0f, 0f, 0f};
        Gemma3ReferenceMath.rmsNormZeroCentered(v, wOne, 1e-6f);
        Gemma3ReferenceMath.rmsNormZeroCentered(vCopy, wZero, 1e-6f);
        for (int i = 0; i < v.length; i++) {
            assertEquals(2.0 * vCopy[i], v[i], 1e-5, "i=" + i);
        }
    }

    @Test
    void geluTanhKnownValues() {
        assertEquals(0.0f, Gemma3ReferenceMath.geluTanh(0f), 1e-6);
        assertEquals(0.841192f, Gemma3ReferenceMath.geluTanh(1f), 1e-4);  // gelu_tanh(1) ~ 0.8412
        assertTrue(Gemma3ReferenceMath.geluTanh(8f) > 7.99f);             // ~identity for large x
        assertTrue(Gemma3ReferenceMath.geluTanh(-8f) > -1e-3f && Gemma3ReferenceMath.geluTanh(-8f) <= 0f);
    }

    @Test
    void ropeAtPositionZeroIsIdentityAndPreservesNorm() {
        float[] v = {0.3f, -0.7f, 1.1f, 0.2f};
        float[] orig = v.clone();
        Gemma3ReferenceMath.applyRopeHalf(v, 0, 4, 0, 10_000.0); // pos 0 -> cos=1,sin=0
        for (int i = 0; i < v.length; i++) {
            assertEquals(orig[i], v[i], 1e-6, "pos0 identity i=" + i);
        }
        // Non-zero position rotates but preserves L2 norm.
        Gemma3ReferenceMath.applyRopeHalf(v, 0, 4, 5, 10_000.0);
        double n1 = norm(orig), n2 = norm(v);
        assertEquals(n1, n2, 1e-4);
    }

    @Test
    void dualRopeThetaProducesDifferentRotations() {
        // headDim 4 -> the i=1 frequency (indices 1 and 3) is theta-dependent (1/sqrt(theta)).
        float[] local = {0f, 1f, 0f, 0f};
        float[] global = {0f, 1f, 0f, 0f};
        Gemma3ReferenceMath.applyRopeHalf(local, 0, 4, 3, 10_000.0);
        Gemma3ReferenceMath.applyRopeHalf(global, 0, 4, 3, 1_000_000.0);
        assertTrue(Math.abs(local[3] - global[3]) > 1e-4, "local vs global theta must differ");
    }

    @Test
    void geGluGateTimesUp() {
        float[] gate = {1f, -8f, 0f};
        float[] up = {2f, 3f, 5f};
        Gemma3ReferenceMath.geluTanhInPlace(gate);
        Gemma3ReferenceMath.multiplyInPlace(gate, up);
        assertEquals(0.841192f * 2f, gate[0], 1e-3);
        assertEquals(0f, gate[2], 1e-6);  // gelu(0)*5 = 0
    }

    private static double norm(float[] a) {
        double s = 0;
        for (float f : a) s += f * f;
        return Math.sqrt(s);
    }
}
