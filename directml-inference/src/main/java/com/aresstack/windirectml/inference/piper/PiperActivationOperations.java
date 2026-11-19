package com.aresstack.windirectml.inference.piper;

/** Implement the nonlinear activations used by Piper VITS as CPU parity references. */
public final class PiperActivationOperations {

    private PiperActivationOperations() {
    }

    public static PiperTensor softplus(PiperTensor input) {
        float[] data = input.requireFloat();
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            double value = data[i];
            result[i] = (float) (Math.max(value, 0d) + Math.log1p(Math.exp(-Math.abs(value))));
        }
        return PiperTensor.floats(result, input.shape());
    }

    public static PiperTensor leakyRelu(PiperTensor input, float slope) {
        if (!Float.isFinite(slope)) {
            throw new IllegalArgumentException("LeakyRelu slope must be finite");
        }
        float[] data = input.requireFloat();
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = data[i] >= 0 ? data[i] : data[i] * slope;
        }
        return PiperTensor.floats(result, input.shape());
    }

    public static PiperTensor ceil(PiperTensor input) {
        float[] data = input.requireFloat();
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (float) Math.ceil(data[i]);
        }
        return PiperTensor.floats(result, input.shape());
    }

    public static PiperTensor clip(PiperTensor input, PiperTensor min, PiperTensor max) {
        int length = input.length();
        validateBound(input, min);
        validateBound(input, max);
        if (input.dataType() == PiperTensor.FLOAT32) {
            float[] source = input.requireFloat();
            float[] result = new float[length];
            for (int i = 0; i < length; i++) {
                float lower = min == null ? Float.NEGATIVE_INFINITY : min.requireFloat()[boundIndex(min, i)];
                float upper = max == null ? Float.POSITIVE_INFINITY : max.requireFloat()[boundIndex(max, i)];
                if (lower > upper) {
                    throw new IllegalArgumentException("Clip lower bound exceeds upper bound");
                }
                result[i] = Math.max(lower, Math.min(upper, source[i]));
            }
            return PiperTensor.floats(result, input.shape());
        }
        long[] source = input.requireInteger();
        long[] result = new long[length];
        for (int i = 0; i < length; i++) {
            long lower = min == null ? Long.MIN_VALUE : min.requireInteger()[boundIndex(min, i)];
            long upper = max == null ? Long.MAX_VALUE : max.requireInteger()[boundIndex(max, i)];
            if (lower > upper) {
                throw new IllegalArgumentException("Clip lower bound exceeds upper bound");
            }
            result[i] = Math.max(lower, Math.min(upper, source[i]));
        }
        return PiperTensor.integers(result, input.shape());
    }

    private static int boundIndex(PiperTensor bound, int position) {
        return bound.length() == 1 ? 0 : position;
    }

    private static void validateBound(PiperTensor source, PiperTensor bound) {
        if (bound == null) {
            return;
        }
        if (source.dataType() != bound.dataType()
                || (bound.length() != 1 && !java.util.Arrays.equals(bound.shape(), source.shape()))) {
            throw new IllegalArgumentException("Clip bound must be scalar or match input shape and type");
        }
    }
}
