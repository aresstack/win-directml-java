package com.aresstack.windirectml.inference.simd;

/**
 * Small float math hot-path abstraction (dot / axpy) used by the reference/CPU paths.
 *
 * <p>This interface has <b>no</b> dependency on {@code jdk.incubator.vector}, so any always-loaded class can use it
 * without the incubator module being present. The SIMD implementation that does depend on the Vector API is isolated
 * behind {@link SimdMath} and loaded reflectively; if the module is absent it is silently replaced by the always
 * available {@link ScalarFloatMathOps}.</p>
 */
public interface FloatMathOps {

    /** Whether this implementation uses the SIMD (Vector API) path. {@code false} for the scalar fallback. */
    boolean enabled();

    /** Dot product of {@code a[aOff..aOff+len)} and {@code b[bOff..bOff+len)}. */
    float dot(float[] a, int aOff, float[] b, int bOff, int len);

    /** {@code out[outOff..outOff+len) += scale * v[vOff..vOff+len)}. */
    void axpy(float[] out, int outOff, float scale, float[] v, int vOff, int len);
}
