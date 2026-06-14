package com.aresstack.windirectml.inference.gemma;

/**
 * Gemma 3-specific scalar math for the device-free CPU reference path (parity oracle for WARP).
 *
 * <p>Differences from the shared {@code DecoderOnlyMath} (SwiGLU/standard RMSNorm):</p>
 * <ul>
 *   <li><b>zero-centered RMSNorm</b>: {@code out = x * rsqrt(mean(x^2)+eps) * (1 + weight)};</li>
 *   <li><b>GELU-tanh</b> activation (gelu_pytorch_tanh) for the GeGLU MLP;</li>
 *   <li><b>QK-norm</b>: zero-centered RMSNorm applied per head over {@code head_dim};</li>
 *   <li><b>RoPE (rotate-half)</b> with a per-layer base frequency (dual local/global theta).</li>
 * </ul>
 */
public final class Gemma3ReferenceMath {

    private static final double GELU_TANH_C = Math.sqrt(2.0 / Math.PI); // ~0.7978845608

    private Gemma3ReferenceMath() {
    }

    /**
     * In-place zero-centered RMSNorm over {@code [offset, offset+length)} of {@code values}:
     * {@code v_i = v_i * rsqrt(mean(v^2)+eps) * (1 + weight_i)} (Gemma scales by {@code 1+weight}).
     */
    public static void rmsNormZeroCentered(float[] values, int offset, int length, float[] weight, float eps) {
        if (weight.length != length) {
            throw new IllegalArgumentException("weight length must equal slice length");
        }
        if (offset < 0 || offset + length > values.length) {
            throw new IllegalArgumentException("slice out of range: offset=" + offset + ", length=" + length);
        }
        double sumSq = 0;
        for (int i = 0; i < length; i++) {
            double v = values[offset + i];
            sumSq += v * v;
        }
        float rms = (float) (1.0 / Math.sqrt(sumSq / length + eps));
        for (int i = 0; i < length; i++) {
            values[offset + i] = (float) (values[offset + i] * rms * (1.0 + weight[i]));
        }
    }

    public static void rmsNormZeroCentered(float[] values, float[] weight, float eps) {
        rmsNormZeroCentered(values, 0, values.length, weight, eps);
    }

    /** GELU with the tanh approximation (gelu_pytorch_tanh): {@code 0.5 x (1 + tanh(c (x + 0.044715 x^3)))}. */
    public static float geluTanh(float x) {
        double inner = GELU_TANH_C * (x + 0.044715 * x * x * x);
        return (float) (0.5 * x * (1.0 + Math.tanh(inner)));
    }

    /** Apply {@link #geluTanh} elementwise, in place. */
    public static void geluTanhInPlace(float[] a) {
        for (int i = 0; i < a.length; i++) {
            a[i] = geluTanh(a[i]);
        }
    }

    /**
     * Apply rotary position embedding (rotate-half / GPT-NeoX style) to a {@code headDim}-wide head
     * vector at {@code [offset, offset+headDim)} for sequence position {@code pos} with base {@code theta}.
     * {@code headDim} must be even.
     */
    public static void applyRopeHalf(float[] vec, int offset, int headDim, int pos, double theta) {
        if ((headDim & 1) != 0) {
            throw new IllegalArgumentException("headDim must be even: " + headDim);
        }
        if (offset < 0 || offset + headDim > vec.length) {
            throw new IllegalArgumentException("rope slice out of range");
        }
        int half = headDim / 2;
        for (int i = 0; i < half; i++) {
            double invFreq = 1.0 / Math.pow(theta, (2.0 * i) / headDim);
            double angle = pos * invFreq;
            double cos = Math.cos(angle);
            double sin = Math.sin(angle);
            float x1 = vec[offset + i];
            float x2 = vec[offset + i + half];
            vec[offset + i] = (float) (x1 * cos - x2 * sin);
            vec[offset + i + half] = (float) (x2 * cos + x1 * sin);
        }
    }

    /** Elementwise {@code dst[i] *= src[i]} (the GeGLU gate*up step). */
    public static void multiplyInPlace(float[] dst, float[] src) {
        if (dst.length != src.length) {
            throw new IllegalArgumentException("length mismatch");
        }
        for (int i = 0; i < dst.length; i++) {
            dst[i] *= src[i];
        }
    }
}
