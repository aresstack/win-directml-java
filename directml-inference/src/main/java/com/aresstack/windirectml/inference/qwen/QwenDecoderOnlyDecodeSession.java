package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyDecodeSession;

import java.util.List;
import java.util.Objects;

/**
 * Experimental {@link DecoderOnlyDecodeSession} backed by Qwen's existing decode pipeline.
 *
 * <p>Drives the real Qwen runtime through the {@link QwenDecodeSteps} seam: {@link #prefill(List)} resets the decode
 * state and runs Qwen's prompt prefill, {@link #decodeNext(int)} runs Qwen's single-token decode. Qwen keeps its own
 * (optionally GPU-resident) KV cache — this session never materialises a CPU {@code DecoderOnlyWarpKvCache}, so the
 * production residency is preserved. Single-use and not thread-safe (it shares the runtime's internal decode state).</p>
 */
final class QwenDecoderOnlyDecodeSession implements DecoderOnlyDecodeSession {

    private final QwenDecodeSteps decodeSteps;
    private boolean prefilled;
    private boolean closed;

    QwenDecoderOnlyDecodeSession(QwenDecodeSteps decodeSteps) {
        this.decodeSteps = Objects.requireNonNull(decodeSteps, "decodeSteps");
    }

    @Override
    public float[] prefill(List<Integer> promptTokenIds) {
        ensureOpen();
        Objects.requireNonNull(promptTokenIds, "promptTokenIds");
        if (prefilled) {
            throw new IllegalStateException("prefill has already been called on this session");
        }
        prefilled = true;
        decodeSteps.resetDecodeState();
        return decodeSteps.decodePrefill(toIntArray(promptTokenIds));
    }

    @Override
    public float[] decodeNext(int tokenId) {
        ensureOpen();
        if (!prefilled) {
            throw new IllegalStateException("prefill must be called before decodeNext");
        }
        return decodeSteps.decodeNextToken(tokenId);
    }

    @Override
    public long lastCallLmHeadNanos() {
        // Qwen's runtime does not split out LM-head timing; report 0 so the loop folds it into the forward time.
        return 0L;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        // Release this run's decode state from the shared runtime; the runtime itself is owned elsewhere.
        decodeSteps.resetDecodeState();
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("decode session is closed");
        }
    }

    private static int[] toIntArray(List<Integer> ids) {
        int[] array = new int[ids.size()];
        for (int i = 0; i < array.length; i++) {
            array[i] = ids.get(i);
        }
        return array;
    }
}
