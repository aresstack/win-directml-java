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
 */
public final class VectorFloatMathOps implements FloatMathOps {

    private final boolean enabled;
    private final VectorSpecies<Float> species;
    private final int lanes;

    public VectorFloatMathOps() {
        boolean ok;
        VectorSpecies<Float> sp = null;
        int l = 0;
        try {
            sp = FloatVector.SPECIES_PREFERRED;
            l = sp.length();
            ok = l >= 4;
        } catch (Throwable t) {
            ok = false;
        }
        this.enabled = ok;
        this.species = sp;
        this.lanes = l;
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
        FloatVector vsum = FloatVector.zero(species);
        int upper = species.loopBound(len);
        int k = 0;
        for (; k < upper; k += lanes) {
            FloatVector va = FloatVector.fromArray(species, a, aOff + k);
            FloatVector vb = FloatVector.fromArray(species, b, bOff + k);
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
        FloatVector vs = FloatVector.broadcast(species, scale);
        int upper = species.loopBound(len);
        int d = 0;
        for (; d < upper; d += lanes) {
            FloatVector vo = FloatVector.fromArray(species, out, outOff + d);
            FloatVector vv = FloatVector.fromArray(species, v, vOff + d);
            vv.fma(vs, vo).intoArray(out, outOff + d);
        }
        for (; d < len; d++) {
            out[outOff + d] += scale * v[vOff + d];
        }
    }
}
