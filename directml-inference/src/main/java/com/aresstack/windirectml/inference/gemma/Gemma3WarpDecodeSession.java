package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.runtime.DirectMlGpuBatch;
import com.aresstack.windirectml.windows.WarpExecutionContext;
import com.aresstack.windirectml.windows.WarpGpuBuffer;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;

import java.util.Objects;

/**
 * Gemma 3 WARP decode session with a reusable KV cache (GEMMA-WARP-10a): a native prefill followed by
 * single-token {@link #decodeNext} steps that reuse the cached k/v instead of re-running the whole
 * sequence.
 *
 * <p>Prefill and decode share one per-token path ({@link Gemma3WarpLayer#decodeStep}): each token at
 * position {@code pos} appends its k/v to the {@link Gemma3WarpKvCache} and attends over the visible
 * cached history (full vs sliding-window per {@link Gemma3AttentionLayout}, GQA preserved,
 * {@code head_dim} explicit). Because token {@code t} attends to keys {@code [firstValid(t), t]}, the
 * last-token logits match the full-sequence {@link Gemma3WarpForwardPass}. The tied LM head is built
 * lazily on the first {@code logits} call. Not the streaming generator or a fused pipeline (that is
 * GEMMA-WARP-10b / the product path). Requires {@link WindowsBindings#isSupported()}.</p>
 */
public final class Gemma3WarpDecodeSession implements AutoCloseable {

    private final Gemma3WarpWeights weights;
    private final Gemma3Config config;
    private final int hidden;
    private final float eps;
    private final float embeddingScale;

    private final Gemma3WarpKernels kernels;
    private final Gemma3WarpLayer[] layers;
    private final Gemma3WarpKvCache cache;
    private final WindowsBindings wb;
    private Gemma3WarpLmHead lmHead; // lazily built on first logits
    private WarpExecutionContext residentCtx; // lazily built for the resident path (13b-3a)
    private WarpGpuBuffer finalNormBuf;       // resident final-norm weight (13b-3a)
    private boolean closed;

    public Gemma3WarpDecodeSession(WindowsBindings wb, Gemma3WarpWeights weights) throws WindowsNativeException {
        this.wb = Objects.requireNonNull(wb, "wb");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.config = weights.config();
        this.hidden = config.hiddenSize();
        this.eps = (float) config.rmsNormEps();
        this.embeddingScale = (float) config.embeddingScale();
        this.kernels = new Gemma3WarpKernels(wb);
        this.layers = new Gemma3WarpLayer[config.numHiddenLayers()];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Gemma3WarpLayer(wb, config, i, weights.layers()[i], kernels);
        }
        this.cache = new Gemma3WarpKvCache(config.numHiddenLayers(), config.keyValueDim(), 16);
    }

    /** Number of positions currently in the cache (prompt + decoded tokens). */
    public int length() {
        return cache.length();
    }

    /**
     * Run the prompt through all layers, filling the KV cache, and return the vocab-sized logits for the
     * last prompt position. Resets any prior cache state.
     */
    public float[] prefill(int[] promptIds) throws WindowsNativeException {
        ensureOpen();
        if (promptIds == null || promptIds.length == 0) {
            throw new IllegalArgumentException("promptIds must not be empty");
        }
        cache.reset();
        float[] last = null;
        for (int t = 0; t < promptIds.length; t++) {
            last = stepToken(promptIds[t], t);
            cache.commitLength(t + 1);
        }
        return logits(last);
    }

    /** Greedy next token id after prefill. */
    public int prefillNextToken(int[] promptIds) throws WindowsNativeException {
        return DecoderOnlyMath.argmax(prefill(promptIds));
    }

    /**
     * Decode one more token: embed {@code tokenId} at the next position, append its k/v to the cache and
     * return the vocab-sized logits for the following token. Requires a prior {@link #prefill}.
     */
    public float[] decodeNext(int tokenId) throws WindowsNativeException {
        ensureOpen();
        if (cache.length() == 0) {
            throw new IllegalStateException("decodeNext requires a prior prefill");
        }
        int pos = cache.length();
        float[] hiddenVec = stepToken(tokenId, pos);
        cache.commitLength(pos + 1);
        return logits(hiddenVec);
    }

    /** Greedy next token id for {@link #decodeNext}. */
    public int decodeNextToken(int tokenId) throws WindowsNativeException {
        return DecoderOnlyMath.argmax(decodeNext(tokenId));
    }

    // ── Resident path (GEMMA-WARP-13b-3a): same math, intermediates stay GPU-resident ──────────────

    /** Resident prefill — same result as {@link #prefill}, far fewer readbacks (only k/v cache + logits). */
    public float[] prefillResident(int[] promptIds) throws WindowsNativeException {
        ensureOpen();
        if (promptIds == null || promptIds.length == 0) {
            throw new IllegalArgumentException("promptIds must not be empty");
        }
        cache.reset();
        WarpGpuBuffer last = null;
        for (int t = 0; t < promptIds.length; t++) {
            if (last != null) {
                last.close();
            }
            last = stepTokenResident(promptIds[t], t);
            cache.commitLength(t + 1);
        }
        return residentLogits(last);
    }

    public int prefillNextTokenResident(int[] promptIds) throws WindowsNativeException {
        return DecoderOnlyMath.argmax(prefillResident(promptIds));
    }

    /** Resident single-token decode — same result as {@link #decodeNext}. */
    public float[] decodeNextResident(int tokenId) throws WindowsNativeException {
        ensureOpen();
        if (cache.length() == 0) {
            throw new IllegalStateException("decodeNextResident requires a prior prefill");
        }
        int pos = cache.length();
        WarpGpuBuffer hidden = stepTokenResident(tokenId, pos);
        cache.commitLength(pos + 1);
        return residentLogits(hidden);
    }

    public int decodeNextTokenResident(int tokenId) throws WindowsNativeException {
        return DecoderOnlyMath.argmax(decodeNextResident(tokenId));
    }

    private WarpExecutionContext ctx() {
        if (residentCtx == null) {
            residentCtx = new WarpExecutionContext(wb);
        }
        return residentCtx;
    }

    private WarpGpuBuffer stepTokenResident(int tokenId, int pos) throws WindowsNativeException {
        if (tokenId < 0 || tokenId >= config.vocabSize()) {
            throw new IllegalArgumentException("token id out of range: " + tokenId);
        }
        int[] one = {tokenId};
        float[] embedRow = weights.hasByteBufferEmbedding()
                ? Gemma3WarpEmbedding.lookupScaled(weights.embeddingFp32Le(), one, hidden, embeddingScale)[0]
                : Gemma3WarpEmbedding.lookupScaled(weights.embeddingFloat(), one, hidden, embeddingScale)[0];
        WarpGpuBuffer hiddenB = ctx().upload(embedRow);
        for (Gemma3WarpLayer layer : layers) {
            WarpGpuBuffer next = layer.decodeStepResident(ctx(), hiddenB, pos, cache);
            hiddenB.close();
            hiddenB = next;
        }
        return hiddenB;
    }

    private float[] residentLogits(WarpGpuBuffer finalHidden) throws WindowsNativeException {
        if (finalNormBuf == null) {
            finalNormBuf = ctx().upload(weights.finalNorm());
        }
        // GEMMA-WARP-13b-3b: coalesce the final-norm dispatch + tied LM-head matvec under one batch; the
        // logits readback drains it (logits must reach the CPU for token selection — the one readback).
        // finalHidden feeds the deferred final-norm dispatch, so it must stay alive until that work has
        // been consumed (the readback drain) — close it only after logits() returns.
        try (DirectMlGpuBatch batch = DirectMlGpuBatch.begin(wb)) {
            WarpGpuBuffer normed = kernels.rmsNorm().normalize(ctx(), finalHidden, finalNormBuf, eps);
            try {
                return lmHead().logits(ctx(), normed);
            } finally {
                normed.close();
                finalHidden.close();
            }
        }
    }

    private float[] stepToken(int tokenId, int pos) throws WindowsNativeException {
        if (tokenId < 0 || tokenId >= config.vocabSize()) {
            throw new IllegalArgumentException("token id out of range: " + tokenId);
        }
        int[] one = {tokenId};
        float[] hiddenVec = weights.hasByteBufferEmbedding()
                ? Gemma3WarpEmbedding.lookupScaled(weights.embeddingFp32Le(), one, hidden, embeddingScale)[0]
                : Gemma3WarpEmbedding.lookupScaled(weights.embeddingFloat(), one, hidden, embeddingScale)[0];
        for (Gemma3WarpLayer layer : layers) {
            hiddenVec = layer.decodeStep(hiddenVec, pos, cache);
        }
        return hiddenVec;
    }

    private float[] logits(float[] lastHidden) throws WindowsNativeException {
        float[] normed = kernels.rmsNorm().normalize(lastHidden, weights.finalNorm(), eps);
        return lmHead().logits(normed);
    }

    private Gemma3WarpLmHead lmHead() {
        if (lmHead == null) {
            lmHead = weights.hasByteBufferEmbedding()
                    ? Gemma3WarpLmHead.fromFp32ByteBuffer(wb, config.vocabSize(), hidden, weights.embeddingFp32Le())
                    : Gemma3WarpLmHead.fromFloatArray(wb, config.vocabSize(), hidden, weights.embeddingFloat());
        }
        return lmHead;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpDecodeSession is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            for (Gemma3WarpLayer layer : layers) {
                layer.close();
            }
            kernels.close();
            if (lmHead != null) {
                lmHead.close();
            }
            if (finalNormBuf != null) {
                finalNormBuf.close();
            }
        }
    }
}
