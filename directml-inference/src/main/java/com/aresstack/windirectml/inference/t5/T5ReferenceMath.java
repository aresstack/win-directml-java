package com.aresstack.windirectml.inference.t5;

import java.util.Arrays;

/**
 * Small reference math helpers for the T5 CPU correctness pipeline.
 *
 * <p>Keep this package-private and test-oriented. Do not route production T5
 * execution through these helpers once the WARP pipeline exists.</p>
 */
final class T5ReferenceMath {
    private T5ReferenceMath() {
    }

    static float[] copy(float[] values) {
        return Arrays.copyOf(values, values.length);
    }

    static void addInPlace(float[] target, float[] residual) {
        requireSameLength(target, residual, "addInPlace");
        for (int i = 0; i < target.length; i++) {
            target[i] += residual[i];
        }
    }

    static float[] add(float[] left, float[] right) {
        requireSameLength(left, right, "add");
        float[] result = new float[left.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = left[i] + right[i];
        }
        return result;
    }

    static float[] dense(float[] input, T5TensorData weight) {
        float[] result = new float[weight.dim(0)];
        denseInto(input, weight, result);
        return result;
    }

    static void denseInto(float[] input, T5TensorData weight, float[] output) {
        if (weight.rank() != 2) {
            throw new IllegalArgumentException("Dense weight must be rank 2: " + weight.name());
        }
        int out = weight.dim(0);
        int in = weight.dim(1);
        if (input.length != in) {
            throw new IllegalArgumentException("Dense input length mismatch for " + weight.name()
                    + ": input=" + input.length + ", expected=" + in);
        }
        if (output.length < out) {
            throw new IllegalArgumentException("Dense output buffer too small for " + weight.name()
                    + ": output=" + output.length + ", expected>=" + out);
        }
        for (int row = 0; row < out; row++) {
            float sum = 0.0f;
            for (int col = 0; col < in; col++) {
                sum += sanitize(input[col]) * sanitize(weight.at(row, col));
            }
            output[row] = finite(sum);
        }
    }

    static float[] denseSequence(float[] input, int sequenceLength, int inputSize, T5TensorData weight) {
        int outputSize = weight.dim(0);
        float[] result = new float[sequenceLength * outputSize];
        for (int token = 0; token < sequenceLength; token++) {
            float[] projected = dense(slice(input, token * inputSize, inputSize), weight);
            System.arraycopy(projected, 0, result, token * outputSize, outputSize);
        }
        return result;
    }

    static float[] slice(float[] source, int offset, int length) {
        float[] result = new float[length];
        System.arraycopy(source, offset, result, 0, length);
        return result;
    }

    static void copyInto(float[] source, float[] target, int targetOffset) {
        System.arraycopy(source, 0, target, targetOffset, source.length);
    }

    static void softmaxInPlace(float[] values) {
        float max = Float.NEGATIVE_INFINITY;
        for (float value : values) {
            if (value > max) {
                max = value;
            }
        }
        double sum = 0.0d;
        for (int i = 0; i < values.length; i++) {
            double exp = Math.exp(values[i] - max);
            values[i] = (float) exp;
            sum += exp;
        }
        if (sum == 0.0d || Double.isNaN(sum) || Double.isInfinite(sum)) {
            float uniform = 1.0f / values.length;
            Arrays.fill(values, uniform);
            return;
        }
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) (values[i] / sum);
        }
    }

    static float relu(float value) {
        return Math.max(0.0f, value);
    }

    static float gelu(float value) {
        double x = value;
        double inner = Math.sqrt(2.0d / Math.PI) * (x + 0.044715d * x * x * x);
        return finite((float) (0.5d * x * (1.0d + Math.tanh(inner))));
    }

    static float finite(float value) {
        if (Float.isNaN(value)) {
            return 0.0f;
        }
        if (value == Float.POSITIVE_INFINITY) {
            return Float.MAX_VALUE;
        }
        if (value == Float.NEGATIVE_INFINITY) {
            return -Float.MAX_VALUE;
        }
        return value;
    }

    private static float sanitize(float value) {
        if (Float.isNaN(value) || Float.isInfinite(value)) {
            return 0.0f;
        }
        return value;
    }

    private static void requireSameLength(float[] left, float[] right, String operation) {
        if (left.length != right.length) {
            throw new IllegalArgumentException(operation + " length mismatch: " + left.length + " != " + right.length);
        }
    }
}
