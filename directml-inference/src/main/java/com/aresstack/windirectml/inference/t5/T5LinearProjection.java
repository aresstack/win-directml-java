package com.aresstack.windirectml.inference.t5;

/**
 * Linear projection used by T5 encoder/decoder blocks.
 *
 * <p>The interface is intentionally tiny so reference projections and
 * WARP/DirectML-backed projections can be swapped without changing the T5
 * encoder/decoder flow. Implementations must compute {@code y = W * x} for a
 * rank-2 weight matrix shaped {@code [output, input]}.</p>
 */
public interface T5LinearProjection extends AutoCloseable {
    /**
     * Return the human-readable projection name.
     *
     * @return projection name
     */
    String name();

    /**
     * Return the expected input size.
     *
     * @return input vector length
     */
    int inputSize();

    /**
     * Return the output size.
     *
     * @return output vector length
     */
    int outputSize();

    /**
     * Apply the projection to one vector.
     *
     * @param input input vector
     * @return projected vector
     */
    float[] apply(float[] input);

    /**
     * Apply the projection to a sequence of row-major vectors.
     *
     * @param input row-major input sequence
     * @param sequenceLength number of tokens
     * @param inputSize input vector length
     * @return row-major output sequence
     */
    default float[] applySequence(float[] input, int sequenceLength, int inputSize) {
        if (inputSize != inputSize()) {
            throw new IllegalArgumentException("Projection input size mismatch for " + name()
                    + ": input=" + inputSize + ", expected=" + inputSize());
        }
        if (input.length != sequenceLength * inputSize) {
            throw new IllegalArgumentException("Projection sequence length mismatch for " + name()
                    + ": values=" + input.length + ", expected=" + (sequenceLength * inputSize));
        }
        float[] result = new float[sequenceLength * outputSize()];
        for (int token = 0; token < sequenceLength; token++) {
            float[] projected = apply(T5ReferenceMath.slice(input, token * inputSize, inputSize));
            T5ReferenceMath.copyInto(projected, result, token * outputSize());
        }
        return result;
    }

    @Override
    default void close() {
        // Keep reference projections no-op closeable.
    }
}
