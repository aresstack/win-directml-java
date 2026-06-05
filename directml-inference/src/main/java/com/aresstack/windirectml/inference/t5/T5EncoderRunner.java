package com.aresstack.windirectml.inference.t5;

/**
 * Execution boundary for the T5 encoder stage.
 *
 * <p>The generation loop depends on this boundary instead of depending on the
 * reference encoder implementation directly. This lets the T5 family replace
 * encoder execution with WARP/DirectML operators later without routing through
 * decoder-only Qwen/SmolLM2 infrastructure.</p>
 */
public interface T5EncoderRunner {

    /**
     * Encode token ids using the runner's default attention-mask policy.
     *
     * @param inputTokenIds source token ids
     * @return encoder output consumed by T5 decoder cross-attention
     */
    T5EncoderOutput encode(int[] inputTokenIds);

    /**
     * Encode token ids with an explicit attention mask.
     *
     * @param inputTokenIds source token ids
     * @param attentionMask true for visible source tokens, false for padding
     * @return encoder output consumed by T5 decoder cross-attention
     */
    T5EncoderOutput encode(int[] inputTokenIds, boolean[] attentionMask);

    /**
     * Describe the execution path used by this runner.
     *
     * @return stable diagnostic execution mode
     */
    String executionMode();
}
