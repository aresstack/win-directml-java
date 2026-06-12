package com.aresstack.windirectml.inference.decoderonly;

/**
 * Long-lived decoder-only forward pass that vends per-run {@link DecoderOnlyDecodeSession}s.
 *
 * <p>Abstracts the forward pass so the {@link DecoderOnlyGenerationLoop} depends on behaviour, not on the concrete
 * GPU-backed {@link DecoderOnlyWarpForwardPass}. The loop never owns or passes a KV cache: it asks for a decode
 * session ({@link #newDecodeSession(int)}) and drives it. This keeps the loop unit-testable without a WARP device and
 * lets a family with a GPU-resident KV cache (e.g. Qwen) plug in without surrendering that residency.</p>
 */
public interface DecoderOnlyForwardPass {

    /** Family-neutral shape view. */
    DecoderOnlyConfig config();

    /** Decode timing accumulator (enabled flag + label decided by the family); long-lived across runs. */
    DecoderOnlyWarpDecodeProfile decodeProfile();

    /**
     * Open a fresh decode session that owns its own decode state / KV cache.
     *
     * @param maxTokens upper bound on prompt + generated tokens for this run (used to size the session's cache)
     */
    DecoderOnlyDecodeSession newDecodeSession(int maxTokens);
}
