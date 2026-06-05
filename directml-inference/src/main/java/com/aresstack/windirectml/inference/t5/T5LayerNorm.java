package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * T5-style bias-free layer normalization for the reference encoder pipeline.
 */
public final class T5LayerNorm {
    private final T5TensorData weight;
    private final float epsilon;

    public T5LayerNorm(T5TensorData weight, float epsilon) {
        this.weight = Objects.requireNonNull(weight, "weight");
        this.epsilon = epsilon <= 0.0f ? 1e-6f : epsilon;
    }

    public float[] apply(float[] input) {
        if (input.length != weight.elementCount()) {
            throw new IllegalArgumentException("LayerNorm input length mismatch for " + weight.name()
                    + ": input=" + input.length + ", expected=" + weight.elementCount());
        }
        double meanSquare = 0.0d;
        for (float value : input) {
            meanSquare += value * value;
        }
        meanSquare /= input.length;
        double scale = 1.0d / Math.sqrt(meanSquare + epsilon);
        float[] result = new float[input.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = T5ReferenceMath.finite((float) (input[i] * scale * weight.at(i)));
        }
        return result;
    }

    public float[] applySequence(float[] input, int sequenceLength, int hiddenSize) {
        float[] result = new float[input.length];
        for (int token = 0; token < sequenceLength; token++) {
            float[] normalized = apply(T5ReferenceMath.slice(input, token * hiddenSize, hiddenSize));
            T5ReferenceMath.copyInto(normalized, result, token * hiddenSize);
        }
        return result;
    }
}
