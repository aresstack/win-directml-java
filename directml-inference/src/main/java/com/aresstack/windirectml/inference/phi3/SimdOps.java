package com.aresstack.windirectml.inference.phi3;

import com.aresstack.windirectml.inference.simd.FloatMathOps;
import com.aresstack.windirectml.inference.simd.SimdMath;

/**
 * SIMD-accelerated dot-product / FMA helpers for the Phi-3 CPU paths.
 *
 * <p>Thin facade over the isolated {@link FloatMathOps} provider ({@link SimdMath}) — the Phi-3-package counterpart of
 * {@code qwen.SimdOps}. No direct {@code jdk.incubator.vector} import, so it loads without the incubator module
 * (scalar fallback then), and uses the Vector API when the module is present (identical results to before).</p>
 */
final class SimdOps {

    private static final FloatMathOps OPS = SimdMath.provider();

    private SimdOps() {
    }

    /** Returns true if the SIMD path is available. */
    static boolean enabled() {
        return OPS.enabled();
    }

    /** Compute the dot product of {@code a[aOff..aOff+len-1]} and {@code b[bOff..bOff+len-1]}. */
    static float dot(float[] a, int aOff, float[] b, int bOff, int len) {
        return OPS.dot(a, aOff, b, bOff, len);
    }

    /** Compute {@code out[outOff..outOff+len-1] += scale * v[vOff..vOff+len-1]}. */
    static void axpy(float[] out, int outOff, float scale, float[] v, int vOff, int len) {
        OPS.axpy(out, outOff, scale, v, vOff, len);
    }
}
