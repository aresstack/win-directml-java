package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyDecodeSession;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * CPU reference {@link DecoderOnlyDecodeSession} for SmolLM2.
 *
 * <p>Lets the shared {@link com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGenerationLoop} drive the
 * SmolLM2 reference forward pass with the same prefill/decode contract as the WARP path, so the reference family no
 * longer needs its own generation loop. It owns a {@link SmolLM2ReferenceKvCache} and the running full token sequence.</p>
 *
 * <p><b>Important:</b> {@link SmolLM2ReferenceForwardPass#logitsForLastToken(List, SmolLM2ReferenceKvCache)} processes
 * the tokens that are new relative to {@code kvCache.completedTokenCount()}, so the session always passes the
 * <em>full</em> sequence (not just the latest token).</p>
 */
final class SmolLM2ReferenceDecodeSession implements DecoderOnlyDecodeSession {

    private final SmolLM2ReferenceForwardPass forwardPass;
    private final SmolLM2ReferenceKvCache kvCache;
    private final List<Integer> fullTokenIds = new ArrayList<>();
    private long lastCallLmHeadNanos;
    private boolean closed;

    SmolLM2ReferenceDecodeSession(SmolLM2ReferenceForwardPass forwardPass) {
        this.forwardPass = Objects.requireNonNull(forwardPass, "forwardPass");
        this.kvCache = SmolLM2ReferenceKvCache.create(forwardPass.config());
    }

    @Override
    public float[] prefill(List<Integer> promptTokenIds) {
        ensureOpen();
        Objects.requireNonNull(promptTokenIds, "promptTokenIds");
        fullTokenIds.clear();
        fullTokenIds.addAll(promptTokenIds);
        return runForward();
    }

    @Override
    public float[] decodeNext(int tokenId) {
        ensureOpen();
        fullTokenIds.add(tokenId);
        return runForward();
    }

    private float[] runForward() {
        long lmHeadBefore = forwardPass.profile().lmHeadNanos();
        float[] logits = forwardPass.logitsForLastToken(fullTokenIds, kvCache);
        lastCallLmHeadNanos = Math.max(0L, forwardPass.profile().lmHeadNanos() - lmHeadBefore);
        return logits;
    }

    @Override
    public long lastCallLmHeadNanos() {
        return lastCallLmHeadNanos;
    }

    @Override
    public void close() {
        // Idempotent; the reference KV cache holds no native resources and the forward pass is not owned here.
        closed = true;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("SmolLM2 reference decode session is closed");
        }
    }
}
