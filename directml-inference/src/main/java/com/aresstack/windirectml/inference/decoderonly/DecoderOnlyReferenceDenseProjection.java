package com.aresstack.windirectml.inference.decoderonly;

import java.util.Objects;

/**
 * Java reference implementation of a decoder-only dense projection.
 */
public final class DecoderOnlyReferenceDenseProjection implements DecoderOnlyDenseProjection {
    private final String name;
    private final int inputSize;
    private final int outputSize;
    private final float[] weights;

    private DecoderOnlyReferenceDenseProjection(String name, int inputSize, int outputSize, float[] weights) {
        this.name = Objects.requireNonNull(name, "name");
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.weights = weights.clone();
    }

    public static DecoderOnlyReferenceDenseProjection fromRowMajorWeights(String name,
                                                                           int outputSize,
                                                                           int inputSize,
                                                                           float[] weights) {
        validateShape(outputSize, inputSize, weights);
        return new DecoderOnlyReferenceDenseProjection(name, inputSize, outputSize, weights);
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public int inputSize() {
        return inputSize;
    }

    @Override
    public int outputSize() {
        return outputSize;
    }

    @Override
    public float[] project(float[] input) {
        float[] output = new float[outputSize];
        projectInto(input, output);
        return output;
    }

    @Override
    public void projectInto(float[] input, float[] output) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Input length mismatch for " + name
                    + ": input=" + input.length + ", expected=" + inputSize);
        }
        if (output.length < outputSize) {
            throw new IllegalArgumentException("Output buffer too small for " + name
                    + ": output=" + output.length + ", expected at least=" + outputSize);
        }
        DecoderOnlyReferenceDenseOps.multiplyRows(weights, outputSize, inputSize, input, output);
    }

    private static void validateShape(int outputSize, int inputSize, float[] weights) {
        Objects.requireNonNull(weights, "weights");
        if (outputSize < 1) {
            throw new IllegalArgumentException("outputSize must be positive: " + outputSize);
        }
        if (inputSize < 1) {
            throw new IllegalArgumentException("inputSize must be positive: " + inputSize);
        }
        long expected = (long) outputSize * inputSize;
        if (weights.length != expected) {
            throw new IllegalArgumentException("Weight length mismatch: weights=" + weights.length
                    + ", expected=" + expected);
        }
    }
}
