package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * Projects a decoder hidden state into vocabulary logits.
 *
 * <p>The interface keeps token generation independent from the execution
 * backend. The reference implementation uses Java math; the WARP bridge uses
 * DirectML through the existing Windows bindings.</p>
 */
public interface T5LogitProjector {
    /**
     * Project the last decoder hidden state to vocabulary logits.
     *
     * @param decoderHiddenState hidden state for one decoder token
     * @return vocabulary logits
     */
    float[] logits(float[] decoderHiddenState);

    /**
     * Project the last decoder hidden state into a caller-owned logits buffer.
     *
     * <p>This method keeps the generation loop allocation-light for WARP-backed
     * decoding. Implementations should override it when they can write directly
     * into {@code outputLogits}; the default implementation preserves the older
     * allocation-returning contract.</p>
     *
     * @param decoderHiddenState hidden state for one decoder token
     * @param outputLogits       target logits buffer with {@link #vocabularySize()} entries
     */
    default void logitsInto(float[] decoderHiddenState, float[] outputLogits) {
        Objects.requireNonNull(outputLogits, "outputLogits");
        if (outputLogits.length < vocabularySize()) {
            throw new IllegalArgumentException("T5 logits buffer too small: " + outputLogits.length
                    + " < " + vocabularySize());
        }
        float[] projected = logits(decoderHiddenState);
        System.arraycopy(projected, 0, outputLogits, 0, vocabularySize());
    }

    /**
     * Return the number of logits produced by this projector.
     *
     * @return vocabulary size
     */
    int vocabularySize();
}
