package com.aresstack.windirectml.inference.t5;

/**
 * Executes the T5 decoder stage behind a replaceable runtime boundary.
 *
 * <p>The reference implementation is {@link T5DecoderPipeline}. WARP-backed
 * implementations must keep decoder self-attention and cross-attention inside
 * the T5 encoder/decoder model family and must not route through decoder-only
 * Qwen/SmolLM2 infrastructure.</p>
 */
public interface T5DecoderRunner {
    /**
     * Return a human-readable execution mode for diagnostics.
     *
     * @return execution mode label
     */
    String executionMode();

    /**
     * Decode a full decoder prefix against an encoder output.
     *
     * @param decoderInputIds decoder prefix tokens
     * @param encoderOutput   encoder output used by cross-attention
     * @return decoder state for the supplied prefix
     */
    T5DecoderState decode(int[] decoderInputIds, T5EncoderOutput encoderOutput);

    /**
     * Decode a single token using the current decoder cache boundary.
     *
     * @param decoderTokenId current decoder token
     * @param encoderOutput  encoder output used by cross-attention
     * @param cache          decoder cache boundary
     * @return decoder state after the step
     */
    T5DecoderState decodeStep(int decoderTokenId, T5EncoderOutput encoderOutput, T5DecoderCache cache);
}
