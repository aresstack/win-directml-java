package com.aresstack.windirectml.inference.decoderonly;

/**
 * Reusable dense projection boundary for decoder-only model families.
 *
 * <p>Implementations may run on the Java reference path, DirectML/WARP, GPU AUTO, or a later quantized kernel. The
 * interface deliberately avoids Qwen- or SmolLM2-specific names so both families can share the same execution seam.</p>
 */
public interface DecoderOnlyDenseProjection extends AutoCloseable {

    String name();

    int inputSize();

    int outputSize();

    float[] project(float[] input);

    void projectInto(float[] input, float[] output);

    default float[] projectSequence(float[] input, int sequenceLength) {
        if (sequenceLength < 1) {
            throw new IllegalArgumentException("sequenceLength must be positive: " + sequenceLength);
        }
        if (input.length != sequenceLength * inputSize()) {
            throw new IllegalArgumentException("Sequence input length mismatch for " + name()
                    + ": values=" + input.length + ", expected=" + (sequenceLength * inputSize()));
        }
        float[] output = new float[sequenceLength * outputSize()];
        projectSequenceInto(input, sequenceLength, output);
        return output;
    }

    default void projectSequenceInto(float[] input, int sequenceLength, float[] output) {
        if (sequenceLength < 1) {
            throw new IllegalArgumentException("sequenceLength must be positive: " + sequenceLength);
        }
        if (input.length != sequenceLength * inputSize()) {
            throw new IllegalArgumentException("Sequence input length mismatch for " + name()
                    + ": values=" + input.length + ", expected=" + (sequenceLength * inputSize()));
        }
        if (output.length < sequenceLength * outputSize()) {
            throw new IllegalArgumentException("Sequence output length mismatch for " + name()
                    + ": values=" + output.length + ", expected at least=" + (sequenceLength * outputSize()));
        }
        float[] row = new float[inputSize()];
        float[] rowOutput = new float[outputSize()];
        for (int token = 0; token < sequenceLength; token++) {
            System.arraycopy(input, token * inputSize(), row, 0, inputSize());
            projectInto(row, rowOutput);
            System.arraycopy(rowOutput, 0, output, token * outputSize(), outputSize());
        }
    }

    @Override
    default void close() {
        // Reference implementations do not own native resources.
    }
}
