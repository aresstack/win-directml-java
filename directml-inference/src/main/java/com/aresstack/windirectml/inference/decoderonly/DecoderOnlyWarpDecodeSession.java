package com.aresstack.windirectml.inference.decoderonly;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * {@link DecoderOnlyDecodeSession} for the SmolLM2-style WARP forward pass.
 *
 * <p>Owns a CPU {@link DecoderOnlyWarpKvCache} and the running token sequence, and drives the long-lived
 * {@link DecoderOnlyWarpForwardPass} via its cache-level {@code logitsForLastToken} entry point. This keeps the WARP
 * decode numerics byte-for-byte identical to the previous loop (which created the cache itself and passed the growing
 * token list each step) while moving cache ownership out of the loop and into the session.</p>
 *
 * <p>The forward pass is long-lived and shared across runs, so {@link #close()} only drops this run's decode state; it
 * never closes the forward pass.</p>
 */
public final class DecoderOnlyWarpDecodeSession implements DecoderOnlyDecodeSession {

    private final DecoderOnlyWarpForwardPass forwardPass;
    private final DecoderOnlyWarpKvCache kvCache;
    private final List<Integer> tokens = new ArrayList<>();
    private boolean prefilled;
    private boolean closed;

    DecoderOnlyWarpDecodeSession(DecoderOnlyWarpForwardPass forwardPass, int maxTokens) {
        this.forwardPass = Objects.requireNonNull(forwardPass, "forwardPass");
        this.kvCache = DecoderOnlyWarpKvCache.create(forwardPass.config(), maxTokens);
    }

    @Override
    public float[] prefill(List<Integer> promptTokenIds) {
        ensureOpen();
        Objects.requireNonNull(promptTokenIds, "promptTokenIds");
        if (prefilled) {
            throw new IllegalStateException("prefill has already been called on this session");
        }
        prefilled = true;
        tokens.addAll(promptTokenIds);
        return forwardPass.logitsForLastToken(tokens, kvCache);
    }

    @Override
    public float[] decodeNext(int tokenId) {
        ensureOpen();
        if (!prefilled) {
            throw new IllegalStateException("prefill must be called before decodeNext");
        }
        tokens.add(tokenId);
        return forwardPass.logitsForLastToken(tokens, kvCache);
    }

    @Override
    public long lastCallLmHeadNanos() {
        return forwardPass.lastCallLmHeadNanos();
    }

    @Override
    public void close() {
        closed = true;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("decode session is closed");
        }
    }
}
