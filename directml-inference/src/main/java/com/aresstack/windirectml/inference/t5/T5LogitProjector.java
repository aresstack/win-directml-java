package com.aresstack.windirectml.inference.t5;

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
     * Return the number of logits produced by this projector.
     *
     * @return vocabulary size
     */
    int vocabularySize();
}
