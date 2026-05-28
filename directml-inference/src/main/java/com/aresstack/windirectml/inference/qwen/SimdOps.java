package com.aresstack.windirectml.inference.qwen;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-accelerated dot-product / FMA helpers backed by the Java Vector API
 * (jdk.incubator.vector). Provides 4–16× speedup on AVX2/AVX-512 CPUs.
 *
 * <p>Falls back to scalar arithmetic if the incubator module is unavailable
 * (e.g. when launched without {@code --add-modules=jdk.incubator.vector}).
 */
final class SimdOps {

    private static final boolean ENABLED;
    private static final VectorSpecies<Float> SP;
    private static final int LANES;

    static {
        boolean ok;
        VectorSpecies<Float> sp = null;
        int lanes = 0;
        try {
            sp = FloatVector.SPECIES_PREFERRED;
            lanes = sp.length();
            ok = lanes >= 4;
        } catch (Throwable t) {
            ok = false;
        }
        ENABLED = ok;
        SP = sp;
        LANES = lanes;
    }

    private SimdOps() {
    }

    /**
     * Returns true if SIMD path is available.
     */
    static boolean enabled() {
        return ENABLED;
    }

    /**
     * Compute the dot product of {@code a[aOff..aOff+len-1]} and
     * {@code b[bOff..bOff+len-1]}.
     */
    static float dot(float[] a, int aOff, float[] b, int bOff, int len) {
        if (!ENABLED) {
            return scalarDot(a, aOff, b, bOff, len);
        }
        FloatVector vsum = FloatVector.zero(SP);
        int upper = SP.loopBound(len);
        int k = 0;
        for (; k < upper; k += LANES) {
            FloatVector va = FloatVector.fromArray(SP, a, aOff + k);
            FloatVector vb = FloatVector.fromArray(SP, b, bOff + k);
            vsum = va.fma(vb, vsum);
        }
        float sum = vsum.reduceLanes(VectorOperators.ADD);
        for (; k < len; k++) {
            sum += a[aOff + k] * b[bOff + k];
        }
        return sum;
    }

    /**
     * Compute {@code out[outOff..outOff+len-1] += scale * v[vOff..vOff+len-1]}.
     */
    static void axpy(float[] out, int outOff, float scale, float[] v, int vOff, int len) {
        if (!ENABLED) {
            for (int d = 0; d < len; d++) out[outOff + d] += scale * v[vOff + d];
            return;
        }
        FloatVector vs = FloatVector.broadcast(SP, scale);
        int upper = SP.loopBound(len);
        int d = 0;
        for (; d < upper; d += LANES) {
            FloatVector vo = FloatVector.fromArray(SP, out, outOff + d);
            FloatVector vv = FloatVector.fromArray(SP, v, vOff + d);
            vv.fma(vs, vo).intoArray(out, outOff + d);
        }
        for (; d < len; d++) out[outOff + d] += scale * v[vOff + d];
    }

    private static float scalarDot(float[] a, int aOff, float[] b, int bOff, int len) {
        float sum = 0f;
        for (int k = 0; k < len; k++) sum += a[aOff + k] * b[bOff + k];
        return sum;
    }
}
