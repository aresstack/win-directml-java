package com.aresstack.windirectml.inference.simd;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Device-free behaviour of the isolated SIMD/scalar float math layer. Runs with or without the Vector module: the
 * provider is the Vector impl when the module is present (CI default) and the scalar impl otherwise; both must produce
 * the same results within tolerance.
 */
class FloatMathOpsTest {

    @Test
    void scalarOpsAreCorrect() {
        FloatMathOps scalar = new ScalarFloatMathOps();
        assertFalse(scalar.enabled());
        assertEquals(32.0f, scalar.dot(new float[]{1, 2, 3}, 0, new float[]{4, 5, 6}, 0, 3), 1e-5f);

        float[] y = {1, 1, 1};
        scalar.axpy(y, 0, 2.0f, new float[]{3, 4, 5}, 0, 3);
        assertArrayEquals(new float[]{7, 9, 11}, y, 1e-5f);
    }

    @Test
    void providerIsResolvedAndMatchesScalarWithinTolerance() {
        FloatMathOps provider = SimdMath.provider();
        assertNotNull(provider);
        FloatMathOps scalar = new ScalarFloatMathOps();

        // 17 elements exercises both the SIMD main loop and the scalar tail of the Vector impl.
        float[] a = new float[17];
        float[] b = new float[17];
        for (int i = 0; i < a.length; i++) {
            a[i] = (i % 5) - 2.0f;
            b[i] = 0.25f * i - 1.0f;
        }
        assertEquals(scalar.dot(a, 0, b, 0, a.length), provider.dot(a, 0, b, 0, a.length), 1e-3f);

        float[] yProvider = new float[a.length];
        float[] yScalar = new float[a.length];
        provider.axpy(yProvider, 0, 1.5f, a, 0, a.length);
        scalar.axpy(yScalar, 0, 1.5f, a, 0, a.length);
        assertArrayEquals(yScalar, yProvider, 1e-3f);
    }
}
