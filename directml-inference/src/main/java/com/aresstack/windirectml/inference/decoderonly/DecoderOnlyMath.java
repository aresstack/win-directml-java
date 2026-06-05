package com.aresstack.windirectml.inference.decoderonly;

import java.util.HashSet;
import java.util.Set;

/**
 * Shared scalar math used by decoder-only CPU fallback paths.
 */
public final class DecoderOnlyMath {

    private DecoderOnlyMath() {
    }

    /**
     * Apply fast SiLU/Swish activation: {@code x * sigmoid(x)}.
     */
    public static float fastSilu(float x) {
        if (x >= 10.0f) {
            return x;
        }
        if (x <= -10.0f) {
            return 0.0f;
        }
        int bits = (int) (-x * 12102203.161561485f + 1065353216.0f);
        float expNeg = Float.intBitsToFloat(bits < 0 ? 0 : bits);
        return x / (1.0f + expNeg);
    }

    public static void rmsNorm(float[] values, float[] weight, float eps) {
        if (values.length != weight.length) {
            throw new IllegalArgumentException("values and weight must have the same length");
        }
        float sumSq = 0;
        for (float value : values) {
            sumSq += value * value;
        }
        float rms = (float) (1.0 / Math.sqrt(sumSq / values.length + eps));
        for (int i = 0; i < values.length; i++) {
            values[i] = values[i] * rms * weight[i];
        }
    }

    public static void softmax(float[] values, int length) {
        if (length < 0 || length > values.length) {
            throw new IllegalArgumentException("length out of range: " + length);
        }
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < length; i++) {
            if (values[i] > max) {
                max = values[i];
            }
        }
        float sum = 0;
        for (int i = 0; i < length; i++) {
            values[i] = (float) Math.exp(values[i] - max);
            sum += values[i];
        }
        float invSum = 1.0f / sum;
        for (int i = 0; i < length; i++) {
            values[i] *= invSum;
        }
    }

    public static int argmax(float[] logits) {
        if (logits.length == 0) {
            throw new IllegalArgumentException("logits must not be empty");
        }
        int maxIdx = 0;
        float maxVal = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    /**
     * Apply repetition penalty in-place to all unique generated token logits.
     */
    public static void applyRepetitionPenalty(float[] logits, int[] generatedIds,
                                              int count, float repetitionPenalty) {
        if (repetitionPenalty <= 1.0f || count <= 0) {
            return;
        }
        if (count > generatedIds.length) {
            throw new IllegalArgumentException("count exceeds generatedIds length");
        }
        Set<Integer> seen = new HashSet<>(count * 2);
        for (int i = 0; i < count; i++) {
            int id = generatedIds[i];
            if (id < 0 || id >= logits.length) {
                continue;
            }
            if (!seen.add(id)) {
                continue;
            }
            float value = logits[id];
            logits[id] = value > 0 ? value / repetitionPenalty : value * repetitionPenalty;
        }
    }
}
