package com.aresstack.windirectml.inference.decoderonly;

import java.util.List;

/**
 * A single decoder-only generation run that <b>owns its own decode state / KV cache</b>.
 *
 * <p>This is the cache-ownership-neutral seam the {@link DecoderOnlyGenerationLoop} drives. Unlike the older
 * cache-in-parameter style, the loop never creates or passes a KV cache: the session keeps whatever decode state its
 * family needs internally. SmolLM2 backs it with a CPU {@link DecoderOnlyWarpKvCache}; a family with a GPU-resident
 * KV cache (e.g. Qwen) can implement the same seam without giving up that residency.</p>
 *
 * <p>Usage is strictly {@link #prefill(List)} once, then {@link #decodeNext(int)} zero or more times, then
 * {@link #close()}. A session is single-use and not thread-safe.</p>
 */
public interface DecoderOnlyDecodeSession extends AutoCloseable {

    /**
     * Process the whole prompt and return the logits for its last position. Must be called exactly once, before any
     * {@link #decodeNext(int)} call.
     */
    float[] prefill(List<Integer> promptTokenIds);

    /**
     * Append one already-selected token and return the logits for the next position.
     */
    float[] decodeNext(int tokenId);

    /**
     * LM-head projection time (ns) measured during the most recent {@link #prefill(List)} / {@link #decodeNext(int)}
     * call. The loop reports this separately so the LM-head cost is not folded into prefill/decoder timings.
     */
    long lastCallLmHeadNanos();

    /** Release any decode state held by the session. Does not close the owning forward pass. */
    @Override
    void close();
}
