package com.aresstack.windirectml.inference.simd.vector;

import com.aresstack.windirectml.inference.simd.FloatMathOps;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD implementation of {@link FloatMathOps} backed by the Java Vector API ({@code jdk.incubator.vector}).
 *
 * <p>This is the ONLY production class that imports {@code jdk.incubator.vector}. It is loaded reflectively by
 * {@link com.aresstack.windirectml.inference.simd.SimdMath}; if the incubator module is not on the module path, loading
 * this class fails and the caller falls back to the scalar implementation. The math (FMA dot reduction, broadcast FMA
 * axpy) is identical to the previous in-line SIMD code, so results are unchanged when the module is present.</p>
 *
 * <p>Must have a public no-arg constructor for reflective instantiation.</p>
 *
 * <p>The species must be a {@code static final} constant: only then does C2 compile the vector
 * operations to SIMD instructions. Held in an instance field, every operation ran through the
 * Vector API's allocating Java fallback and was several times slower than scalar code.</p>
 */
public final class VectorFloatMathOps implements FloatMathOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int LANES = SPECIES.length();

    private final boolean enabled = LANES >= 4;

    public VectorFloatMathOps() {
    }

    @Override
    public boolean enabled() {
        return enabled;
    }

    @Override
    public float dot(float[] a, int aOff, float[] b, int bOff, int len) {
        if (!enabled) {
            float sum = 0.0f;
            for (int k = 0; k < len; k++) {
                sum += a[aOff + k] * b[bOff + k];
            }
            return sum;
        }
        FloatVector vsum = FloatVector.zero(SPECIES);
        int upper = SPECIES.loopBound(len);
        int k = 0;
        for (; k < upper; k += LANES) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, aOff + k);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, bOff + k);
            vsum = va.fma(vb, vsum);
        }
        float sum = vsum.reduceLanes(VectorOperators.ADD);
        for (; k < len; k++) {
            sum += a[aOff + k] * b[bOff + k];
        }
        return sum;
    }

    @Override
    public void axpy(float[] out, int outOff, float scale, float[] v, int vOff, int len) {
        if (!enabled) {
            for (int d = 0; d < len; d++) {
                out[outOff + d] += scale * v[vOff + d];
            }
            return;
        }
        FloatVector vs = FloatVector.broadcast(SPECIES, scale);
        int upper = SPECIES.loopBound(len);
        int d = 0;
        for (; d < upper; d += LANES) {
            FloatVector vo = FloatVector.fromArray(SPECIES, out, outOff + d);
            FloatVector vv = FloatVector.fromArray(SPECIES, v, vOff + d);
            vv.fma(vs, vo).intoArray(out, outOff + d);
        }
        for (; d < len; d++) {
            out[outOff + d] += scale * v[vOff + d];
        }
    }
}
