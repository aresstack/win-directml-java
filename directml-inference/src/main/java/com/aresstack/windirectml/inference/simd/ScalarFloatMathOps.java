package com.aresstack.windirectml.inference.simd;

/**
 * Always-loadable scalar implementation of {@link FloatMathOps}. No {@code jdk.incubator.vector} dependency, so it
 * works even when the incubator module is not on the module path. This is the fallback {@link SimdMath} returns when
 * the Vector implementation cannot be loaded.
 */
public final class ScalarFloatMathOps implements FloatMathOps {

    @Override
    public boolean enabled() {
        return false;
    }

    @Override
    public float dot(float[] a, int aOff, float[] b, int bOff, int len) {
        float sum = 0.0f;
        for (int k = 0; k < len; k++) {
            sum += a[aOff + k] * b[bOff + k];
        }
        return sum;
    }

    @Override
    public void axpy(float[] out, int outOff, float scale, float[] v, int vOff, int len) {
        for (int d = 0; d < len; d++) {
            out[outOff + d] += scale * v[vOff + d];
        }
    }
}
