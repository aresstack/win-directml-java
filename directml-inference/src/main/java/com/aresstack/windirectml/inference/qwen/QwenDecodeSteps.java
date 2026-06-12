package com.aresstack.windirectml.inference.qwen;

/**
 * Minimal internal seam exposing Qwen's existing step-wise decode primitives to the experimental
 * decoder-only session adapter.
 *
 * <p>{@link Qwen2Runtime} already decodes step by step internally (prompt prefill, then one token at a time) with its
 * own — optionally GPU-resident — KV cache. This interface surfaces exactly those primitives so
 * {@link QwenDecoderOnlyDecodeSession} can drive the real Qwen pipeline without copying SmolLM2's CPU cache model.
 * The production {@code generateStreaming} path and the underlying private {@code prefill}/{@code decodeSingleToken}
 * methods are unchanged; the implementing runtime only adds thin delegations.</p>
 */
public interface QwenDecodeSteps {

    /** Clear all decode state (KV cache, cached sequence length) for a fresh run. */
    void resetDecodeState();

    /** Process the whole prompt and return logits for its last position. */
    float[] decodePrefill(int[] promptTokenIds);

    /** Decode one already-selected token and return logits for the next position. */
    float[] decodeNextToken(int tokenId);
}
