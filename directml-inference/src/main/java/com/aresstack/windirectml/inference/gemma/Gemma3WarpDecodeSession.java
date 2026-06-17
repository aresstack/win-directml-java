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

    private static final org.slf4j.Logger LOG =
            org.slf4j.LoggerFactory.getLogger(Gemma3WarpDecodeSession.class);

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
    private Gemma3WarpResidentKvCache residentKv; // GPU-resident KV cache for the resident path (13c)
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
        // GEMMA-WARP-16: the projection-fusion state (q/k/v → one DML-GEMM, gate/up → one DML-GEMM). DEBUG
        // so the normal runtime stays quiet (GEMMA-WARP-CLOSEOUT); raise the logger to see it when probing.
        if (LOG.isDebugEnabled()) {
            LOG.debug("Gemma projection fusion: QKV fused active={}, GateUp fused active={}",
                    layers[0].qkvFused(), layers[0].gateUpFused());
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
        residentKv().reset();
        WarpGpuBuffer last = null;
        for (int t = 0; t < promptIds.length; t++) {
            if (last != null) {
                last.close();
            }
            last = stepTokenResident(promptIds[t], t);
            residentKv().commitLength(t + 1);
        }
        return residentLogits(last);
    }

    public int prefillNextTokenResident(int[] promptIds) throws WindowsNativeException {
        return DecoderOnlyMath.argmax(prefillResident(promptIds));
    }

    /**
     * Batched resident prefill (GEMMA-WARP-13e): processes the whole prompt at once — all positions through
     * each layer with batched projections + batched MLP and one GPU-resident KV append per position — instead
     * of the token-by-token {@link #prefillResident}. Same result; far fewer submits for long prompts. Only
     * the last position's logits are computed (the LM head runs once). The token-by-token
     * {@link #prefillResident} stays as the debug/fallback oracle.
     */
    public float[] prefillResidentBatched(int[] promptIds) throws WindowsNativeException {
        ensureOpen();
        if (promptIds == null || promptIds.length == 0) {
            throw new IllegalArgumentException("promptIds must not be empty");
        }
        int seqLen = promptIds.length;
        residentKv().reset();
        residentKv().ensureCapacity(ctx(), seqLen);
        float[][] emb = weights.embedScaled(promptIds, embeddingScale);
        WarpGpuBuffer[] h = new WarpGpuBuffer[seqLen];
        for (int p = 0; p < seqLen; p++) {
            h[p] = ctx().upload(emb[p]);
        }
        for (Gemma3WarpLayer layer : layers) {
            WarpGpuBuffer[] next = layer.forwardPrefillBatched(ctx(), h, residentKv());
            for (WarpGpuBuffer b : h) {
                b.close();
            }
            h = next;
        }
        residentKv().commitLength(seqLen);
        for (int p = 0; p < seqLen - 1; p++) {
            h[p].close();
        }
        return residentLogits(h[seqLen - 1]); // closes the last hidden
    }

    public int prefillNextTokenResidentBatched(int[] promptIds) throws WindowsNativeException {
        return DecoderOnlyMath.argmax(prefillResidentBatched(promptIds));
    }

    /** Resident single-token decode — same result as {@link #decodeNext}. */
    public float[] decodeNextResident(int tokenId) throws WindowsNativeException {
        ensureOpen();
        if (residentKv().length() == 0) {
            throw new IllegalStateException("decodeNextResident requires a prior prefill");
        }
        int pos = residentKv().length();
        WarpGpuBuffer hidden = stepTokenResident(tokenId, pos);
        residentKv().commitLength(pos + 1);
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

    /** Lazily-built GPU-resident KV cache for the native-warp product path (GEMMA-WARP-13c). */
    private Gemma3WarpResidentKvCache residentKv() throws WindowsNativeException {
        if (residentKv == null) {
            residentKv = new Gemma3WarpResidentKvCache(ctx(), config.numHiddenLayers(), config.keyValueDim(), 32);
        }
        return residentKv;
    }

    private WarpGpuBuffer stepTokenResident(int tokenId, int pos) throws WindowsNativeException {
        if (tokenId < 0 || tokenId >= config.vocabSize()) {
            throw new IllegalArgumentException("token id out of range: " + tokenId);
        }
        int[] one = {tokenId};
        float[] embedRow = weights.embedScaled(one, embeddingScale)[0];
        // Grow the resident KV cache for this position OUTSIDE the per-layer batch (growth copies + frees
        // the old buffers synchronously, which must not race with deferred work).
        residentKv().ensureCapacity(ctx(), pos + 1);
        WarpGpuBuffer hiddenB = ctx().upload(embedRow);
        for (Gemma3WarpLayer layer : layers) {
            WarpGpuBuffer next = layer.decodeStepResident(ctx(), hiddenB, pos, residentKv());
            hiddenB.close();
            hiddenB = next;
        }
        return hiddenB;
    }

    private float[] residentLogits(WarpGpuBuffer finalHidden) throws WindowsNativeException {
        if (finalNormBuf == null) {
            finalNormBuf = ctx().upload(weights.finalNorm());
        }
        // GEMMA-WARP-13d: per-layer-style batch (deferred fence) + command-list coalescing for the final
        // norm + tied LM-head matvec; the single logits readback drains the batch (logits must reach the CPU
        // for token selection). finalHidden feeds the recorded final-norm, so it stays alive until the drain.
        // GEMMA-WARP-17: in group-profiling mode run synchronously (no batch/coalescing) so the tail groups
        // (final-rmsnorm, lm-head, logits-readback) are individually timed by the mark(..) boundaries.
        boolean profiling = ctx().isGroupProfiling();
        DirectMlGpuBatch batch = profiling ? null : DirectMlGpuBatch.begin(wb);
        try {
            ctx().beginRecording();
            WarpGpuBuffer normed = null;
            WarpGpuBuffer logitsBuf = null;
            try {
                ctx().mark("final-rmsnorm");
                normed = kernels.rmsNorm().normalize(ctx(), finalHidden, finalNormBuf, eps);
                ctx().mark("lm-head");
                logitsBuf = lmHead().logitsResident(ctx(), normed);
                if (!profiling) {
                    ctx().flushRecording();     // submit pending list (deferred into the batch)
                }
                ctx().mark("logits-readback");
                float[] result = logitsBuf.readback(); // drains the deferred work; the one logits readback
                ctx().endGroup();               // close the tail so token-selection is attributed separately
                return result;
            } finally {
                if (ctx().isRecording()) {
                    ctx().flushRecording();      // close the coalescing scope (both modes)
                }
                if (logitsBuf != null) {
                    logitsBuf.close();
                }
                if (normed != null) {
                    normed.close();
                }
                finalHidden.close();
            }
        } finally {
            if (batch != null) {
                batch.close();
            }
        }
    }

    /**
     * GEMMA-WARP-17: attach (or clear with {@code null}) an opt-in per-group decode timing profiler. While
     * attached, the resident decode/logits path runs synchronously so each kernel group is individually
     * timed (see {@link WarpGroupProfiler}). Measurement-only; the normal runtime never attaches one.
     */
    public void setGroupProfiler(WarpGroupProfiler profiler) {
        ctx().setGroupSink(profiler == null ? null : profiler::record);
    }

    private float[] stepToken(int tokenId, int pos) throws WindowsNativeException {
        if (tokenId < 0 || tokenId >= config.vocabSize()) {
            throw new IllegalArgumentException("token id out of range: " + tokenId);
        }
        int[] one = {tokenId};
        float[] hiddenVec = weights.embedScaled(one, embeddingScale)[0];
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
            lmHead = weights.buildLmHead(wb);
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
            if (residentKv != null) {
                residentKv.close();
            }
        }
    }
}
