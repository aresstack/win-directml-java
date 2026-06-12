package com.aresstack.windirectml.inference.decoderonly;

import com.aresstack.windirectml.windows.GpuPipeline;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Native DirectML/WARP forward pass shared by decoder-only model families (Qwen-like and Llama-like, e.g. SmolLM2).
 *
 * <p>Keeps the canonical decoder-only numerical structure (RMSNorm, RoPE, grouped-query attention, SwiGLU MLP,
 * tied/untied LM head) and runs every dense projection on the shared {@link DecoderOnlyWarpDenseProjection} GPU
 * kernel. Norms, RoPE, attention scoring and the SwiGLU non-linearity use the shared {@code decoderonly} math so the
 * WARP path stays aligned with the CPU reference path apart from GPU floating-point rounding.</p>
 *
 * <p>All family-specific inputs (config view, rotary table, token embedding, per-layer projections, final norm and LM
 * head) are resolved by the model family and handed in; this class owns their lifecycle from construction onward. The
 * MLP block infrastructure (one shared pipeline + SwiGLU kernel, one GPU-resident block per layer) is built here.</p>
 *
 * <p>The projections are uploaded once (by the family) and reused across every decode step. Attention and the KV
 * cache run on the CPU exactly like the reference path; only the matmul-heavy projections move to WARP. The LM head
 * runs on WARP by default ({@code lmHead != null}); the CPU-SIMD reference projection is only used when a family opts
 * into the dev/diagnostic LM-head fallback, never to optimize the WARP path.</p>
 */
public final class DecoderOnlyWarpForwardPass implements AutoCloseable, DecoderOnlyForwardPass {

    private final DecoderOnlyConfig config;
    private final DecoderOnlyAttentionLayout attentionLayout;
    private final DecoderOnlyRotaryEmbedding rotaryEmbedding;
    private final float attentionScale;
    private final float rmsNormEps;

    private final DecoderOnlyEmbeddingTable tokenEmbedding;
    private final float[] finalNorm;
    private final List<DecoderOnlyWarpLayer> layers;
    /** WARP product path for the LM head (default); null only when the dev reference LM head is in use. */
    private final DecoderOnlyWarpDenseProjection lmHead;
    /** CPU-SIMD reference LM head; only set (and only used) when {@link #lmHead} is null. */
    private final DecoderOnlyReferenceProjection lmHeadReference;

    // Decode dimensions (identical across layers for a single decoder-only model).
    private final int hiddenSize;
    private final int qSize;
    private final int kvSize;
    private final int intermediateSize;

    // GPU-resident MLP block (gate_up → WARP SwiGLU → down in one submission) per layer, plus the shared pipeline
    // and SwiGLU kernel they record into. Removes the per-layer gate/up readback + intermediate re-upload.
    private final GpuPipeline mlpPipeline;
    private final DecoderOnlyWarpSwiGluKernel swiGluKernel;
    private final List<DecoderOnlyWarpMlpBlock> mlpBlocks;

    // Reusable per-instance scratch buffers for the single-token decode path. Generation is sequential per session,
    // so these are reused across decode steps instead of allocating fresh arrays every token/layer.
    private final float[] scratchHidden;
    private final float[] scratchAttnNorm;
    private final float[] scratchQkv;
    private final float[] scratchQuery;
    private final float[] scratchKey;
    private final float[] scratchValue;
    private final float[] scratchContext;
    private final float[] scratchAttnOut;
    private final float[] scratchMlpNorm;
    private final float[] scratchGateUp;
    private final float[] scratchGate;
    private final float[] scratchUp;
    private final float[] scratchDown;
    private final float[] scratchFinalNorm;
    private final float[] scratchScores;

    /** Fine-grained decode timings, only populated when the injected profile is enabled. */
    private final DecoderOnlyWarpDecodeProfile decodeProfile;

    /** LM-head projection time accumulated during the most recent {@link #logitsForLastToken} call. */
    private long lastCallLmHeadNanos;

    private boolean closed;

    /**
     * @param windowsBindings native bindings used to build the GPU-resident MLP block infrastructure
     * @param config          family-neutral shape view (dims, GQA head counts, vocab/position bounds, eps)
     * @param rotaryEmbedding rotary table built by the family (passed in to keep {@code rope_theta} at full
     *                        precision; the {@link DecoderOnlyConfig} contract narrows it to {@code float})
     * @param tokenEmbedding  embedding-row lookup seam
     * @param layers          per-layer projections + norm vectors; ownership transfers to this pass on success
     * @param finalNorm       final RMSNorm gain vector [hidden]
     * @param lmHead          WARP LM-head projection (the product path); null iff {@code lmHeadReference} is set
     * @param lmHeadReference CPU reference LM-head projection (dev/diagnostic); null iff {@code lmHead} is set
     * @param decodeProfile   injected timing accumulator (enabled flag + label decided by the family)
     */
    public DecoderOnlyWarpForwardPass(WindowsBindings windowsBindings,
                                      DecoderOnlyConfig config,
                                      DecoderOnlyRotaryEmbedding rotaryEmbedding,
                                      DecoderOnlyEmbeddingTable tokenEmbedding,
                                      List<DecoderOnlyWarpLayer> layers,
                                      float[] finalNorm,
                                      DecoderOnlyWarpDenseProjection lmHead,
                                      DecoderOnlyReferenceProjection lmHeadReference,
                                      DecoderOnlyWarpDecodeProfile decodeProfile) {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.config = Objects.requireNonNull(config, "config");
        this.rotaryEmbedding = Objects.requireNonNull(rotaryEmbedding, "rotaryEmbedding");
        this.tokenEmbedding = Objects.requireNonNull(tokenEmbedding, "tokenEmbedding");
        this.finalNorm = Objects.requireNonNull(finalNorm, "finalNorm");
        Objects.requireNonNull(layers, "layers");
        if (layers.isEmpty()) {
            throw new IllegalArgumentException("layers must not be empty");
        }
        if ((lmHead == null) == (lmHeadReference == null)) {
            throw new IllegalArgumentException("exactly one of lmHead / lmHeadReference must be set");
        }
        this.decodeProfile = Objects.requireNonNull(decodeProfile, "decodeProfile");

        this.attentionLayout = new DecoderOnlyAttentionLayout(
                config.numAttentionHeads(), config.numKeyValueHeads());
        this.attentionScale = (float) (1.0d / Math.sqrt(config.headDim()));
        this.rmsNormEps = config.rmsNormEps();
        this.layers = List.copyOf(layers);
        this.lmHead = lmHead;
        this.lmHeadReference = lmHeadReference;

        this.hiddenSize = config.hiddenSize();
        this.qSize = config.numAttentionHeads() * config.headDim();
        this.kvSize = config.numKeyValueHeads() * config.headDim();
        DecoderOnlyWarpLayer firstLayer = this.layers.get(0);
        this.intermediateSize = firstLayer.gateUpProjection.sliceSize(DecoderOnlyWarpLayer.GATE_UP_GATE);
        int qkvTotal = firstLayer.qkvProjection.totalOutputSize();
        int gateUpTotal = firstLayer.gateUpProjection.totalOutputSize();

        this.scratchHidden = new float[hiddenSize];
        this.scratchAttnNorm = new float[hiddenSize];
        this.scratchQkv = new float[qkvTotal];
        this.scratchQuery = new float[qSize];
        this.scratchKey = new float[kvSize];
        this.scratchValue = new float[kvSize];
        this.scratchContext = new float[hiddenSize];
        this.scratchAttnOut = new float[hiddenSize];
        this.scratchMlpNorm = new float[hiddenSize];
        this.scratchGateUp = new float[gateUpTotal];
        this.scratchGate = new float[intermediateSize];
        this.scratchUp = new float[intermediateSize];
        this.scratchDown = new float[hiddenSize];
        this.scratchFinalNorm = new float[hiddenSize];
        this.scratchScores = new float[config.maxPositionEmbeddings()];

        // GPU-resident MLP block infrastructure: one shared pipeline + SwiGLU kernel, one block per layer.
        // On failure here, only the partial infrastructure is released; the handed-in layers/LM head remain the
        // caller's responsibility until this constructor returns successfully.
        GpuPipeline builtPipeline = null;
        DecoderOnlyWarpSwiGluKernel builtSwiGlu = null;
        try {
            long activationBytes = (long) hiddenSize * Float.BYTES;
            builtPipeline = new GpuPipeline(windowsBindings, activationBytes, activationBytes);
            builtSwiGlu = new DecoderOnlyWarpSwiGluKernel(
                    windowsBindings, builtPipeline.getCommandList(), intermediateSize);
            List<DecoderOnlyWarpMlpBlock> blocks = new ArrayList<>(this.layers.size());
            for (DecoderOnlyWarpLayer layer : this.layers) {
                blocks.add(new DecoderOnlyWarpMlpBlock(builtPipeline,
                        layer.gateUpProjection.kernel(), layer.downProjection.kernel(),
                        builtSwiGlu, hiddenSize));
            }
            this.mlpPipeline = builtPipeline;
            this.swiGluKernel = builtSwiGlu;
            this.mlpBlocks = List.copyOf(blocks);
        } catch (WindowsNativeException | RuntimeException e) {
            if (builtSwiGlu != null) {
                builtSwiGlu.close();
            }
            if (builtPipeline != null) {
                builtPipeline.close();
            }
            throw new RuntimeException("Decoder-only WARP MLP pipeline initialisation failed", e);
        }
    }

    @Override
    public DecoderOnlyConfig config() {
        return config;
    }

    @Override
    public DecoderOnlyWarpDecodeProfile decodeProfile() {
        return decodeProfile;
    }

    @Override
    public DecoderOnlyDecodeSession newDecodeSession(int maxTokens) {
        ensureOpen();
        return new DecoderOnlyWarpDecodeSession(this, maxTokens);
    }

    /**
     * LM-head projection time (nanoseconds) measured during the most recent {@link #logitsForLastToken} call. The
     * generation loop reports this separately so the LM-head cost is not silently folded into prefill/decoder timings.
     */
    public long lastCallLmHeadNanos() {
        return lastCallLmHeadNanos;
    }

    /**
     * Decode the supplied token sequence with an incremental KV cache and return the logits for the last token.
     *
     * <p>When more than one new token must be processed (the initial prompt prefill) the whole new-token block is
     * processed sequence-wise: every dense projection runs once per layer over the full block through
     * {@link DecoderOnlyWarpDenseProjection#projectSequenceInto}, instead of one WARP dispatch per token per
     * projection. This is mathematically identical to the per-token loop (same RMSNorm/RoPE/causal-attention/SwiGLU)
     * but issues far fewer WARP dispatches. The single-token decode step keeps the per-token path.</p>
     */
    public float[] logitsForLastToken(List<Integer> tokenIds, DecoderOnlyWarpKvCache kvCache) {
        ensureOpen();
        Objects.requireNonNull(tokenIds, "tokenIds");
        Objects.requireNonNull(kvCache, "kvCache");
        if (tokenIds.isEmpty()) {
            throw new IllegalArgumentException("tokenIds must not be empty");
        }
        lastCallLmHeadNanos = 0L;
        int startPosition = kvCache.completedTokenCount();
        if (startPosition > tokenIds.size()) {
            throw new IllegalArgumentException("KV cache contains more tokens than the supplied sequence");
        }
        if (startPosition == tokenIds.size()) {
            throw new IllegalStateException("KV cache already contained the full sequence");
        }
        if (tokenIds.size() - startPosition > 1) {
            return prefillSequence(tokenIds, kvCache, startPosition);
        }
        float[] logits = null;
        int lastPosition = tokenIds.size() - 1;
        for (int position = startPosition; position < tokenIds.size(); position++) {
            logits = logitsForToken(tokenIds.get(position), position, kvCache, position == lastPosition);
        }
        if (logits == null) {
            throw new IllegalStateException("KV cache already contained the full sequence");
        }
        return logits;
    }

    private float[] logitsForToken(int tokenId, int position, DecoderOnlyWarpKvCache kvCache, boolean projectLogits) {
        if (tokenId < 0 || tokenId >= config.vocabSize()) {
            throw new IllegalArgumentException("tokenId out of vocabulary range: " + tokenId);
        }
        if (position >= config.maxPositionEmbeddings()) {
            throw new IllegalArgumentException("position exceeds max_position_embeddings: " + position);
        }
        kvCache.requireReadyForPosition(position);
        if (kvCache.layerCount() != layers.size()) {
            throw new IllegalArgumentException("KV cache layer count mismatch: expected "
                    + layers.size() + " but got " + kvCache.layerCount());
        }
        boolean prof = decodeProfile.enabled();
        if (prof) {
            decodeProfile.decodeSteps++;
        }
        long t = prof ? System.nanoTime() : 0L;
        tokenEmbedding.copyRowInto(tokenId, scratchHidden);
        if (prof) {
            decodeProfile.embedding += System.nanoTime() - t;
        }
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            runLayerStepInPlace(scratchHidden, layerIndex, layers.get(layerIndex), kvCache.layer(layerIndex),
                    position, prof);
        }
        if (!projectLogits) {
            return null;
        }
        return projectLogits(scratchHidden);
    }

    /** Apply the final RMSNorm and project the LM head (WARP by default; CPU-SIMD only in the dev reference mode). */
    private float[] projectLogits(float[] hidden) {
        System.arraycopy(hidden, 0, scratchFinalNorm, 0, hiddenSize);
        DecoderOnlyMath.rmsNorm(scratchFinalNorm, finalNorm, rmsNormEps);
        long lmHeadStart = System.nanoTime();
        float[] logits = lmHead != null
                ? lmHead.project(scratchFinalNorm)
                : lmHeadReference.project(scratchFinalNorm);
        long elapsed = System.nanoTime() - lmHeadStart;
        lastCallLmHeadNanos += elapsed;
        if (decodeProfile.enabled()) {
            decodeProfile.lmHead += elapsed;
        }
        return logits;
    }

    private void runLayerStepInPlace(float[] hidden, int layerIndex, DecoderOnlyWarpLayer layer,
                                     DecoderOnlyWarpLayerKvCache layerCache, int position, boolean prof) {
        long t = prof ? System.nanoTime() : 0L;
        System.arraycopy(hidden, 0, scratchAttnNorm, 0, hiddenSize);
        DecoderOnlyMath.rmsNorm(scratchAttnNorm, layer.inputNorm, rmsNormEps);
        if (prof) {
            decodeProfile.attentionNorm += System.nanoTime() - t;
        }

        runSelfAttentionStepInto(scratchAttnNorm, layer, layerCache, position, prof); // writes scratchAttnOut

        t = prof ? System.nanoTime() : 0L;
        for (int i = 0; i < hiddenSize; i++) {
            hidden[i] += scratchAttnOut[i];
        }
        if (prof) {
            decodeProfile.attentionResidual += System.nanoTime() - t;
        }

        t = prof ? System.nanoTime() : 0L;
        System.arraycopy(hidden, 0, scratchMlpNorm, 0, hiddenSize);
        DecoderOnlyMath.rmsNorm(scratchMlpNorm, layer.postAttentionNorm, rmsNormEps);
        if (prof) {
            decodeProfile.mlpNorm += System.nanoTime() - t;
        }

        runMlpStepInto(layerIndex, prof); // writes scratchDown

        t = prof ? System.nanoTime() : 0L;
        for (int i = 0; i < hiddenSize; i++) {
            hidden[i] += scratchDown[i];
        }
        if (prof) {
            decodeProfile.mlpResidual += System.nanoTime() - t;
        }
    }

    private void runSelfAttentionStepInto(float[] input, DecoderOnlyWarpLayer layer,
                                          DecoderOnlyWarpLayerKvCache layerCache, int position, boolean prof) {
        int numHeads = config.numAttentionHeads();
        int numKvHeads = config.numKeyValueHeads();
        int headDim = config.headDim();

        long t = prof ? System.nanoTime() : 0L;
        layer.qkvProjection.projectInto(input, scratchQkv);
        if (prof) {
            decodeProfile.qkvProjection += System.nanoTime() - t;
        }

        t = prof ? System.nanoTime() : 0L;
        layer.qkvProjection.copySlice(scratchQkv, DecoderOnlyWarpLayer.QKV_QUERY, scratchQuery);
        layer.qkvProjection.copySlice(scratchQkv, DecoderOnlyWarpLayer.QKV_KEY, scratchKey);
        layer.qkvProjection.copySlice(scratchQkv, DecoderOnlyWarpLayer.QKV_VALUE, scratchValue);
        if (prof) {
            decodeProfile.qkvSlice += System.nanoTime() - t;
        }

        t = prof ? System.nanoTime() : 0L;
        for (int head = 0; head < numHeads; head++) {
            rotaryEmbedding.apply(scratchQuery, head * headDim, position);
        }
        for (int head = 0; head < numKvHeads; head++) {
            rotaryEmbedding.apply(scratchKey, head * headDim, position);
        }
        if (prof) {
            decodeProfile.rope += System.nanoTime() - t;
        }

        t = prof ? System.nanoTime() : 0L;
        layerCache.append(scratchKey, scratchValue);
        if (prof) {
            decodeProfile.kvAppend += System.nanoTime() - t;
        }

        Arrays.fill(scratchContext, 0, hiddenSize, 0.0f);
        float[] keys = layerCache.keys();
        float[] values = layerCache.values();
        int keyWidth = layerCache.keyWidth();
        int sourceCount = layerCache.size();
        for (int queryHead = 0; queryHead < numHeads; queryHead++) {
            int kvHead = attentionLayout.kvHeadForQueryHead(queryHead);
            int queryOffset = queryHead * headDim;
            int kvHeadOffset = kvHead * headDim;

            long ts = prof ? System.nanoTime() : 0L;
            for (int sourcePosition = 0; sourcePosition < sourceCount; sourcePosition++) {
                scratchScores[sourcePosition] = dotOffset(
                        scratchQuery, queryOffset,
                        keys, sourcePosition * keyWidth + kvHeadOffset,
                        headDim) * attentionScale;
            }
            if (prof) {
                decodeProfile.attentionScore += System.nanoTime() - ts;
            }

            long tsm = prof ? System.nanoTime() : 0L;
            DecoderOnlyMath.softmax(scratchScores, sourceCount);
            if (prof) {
                decodeProfile.softmax += System.nanoTime() - tsm;
            }

            long tc = prof ? System.nanoTime() : 0L;
            for (int sourcePosition = 0; sourcePosition < sourceCount; sourcePosition++) {
                addWeightedOffset(scratchContext, queryOffset,
                        values, sourcePosition * keyWidth + kvHeadOffset, headDim, scratchScores[sourcePosition]);
            }
            if (prof) {
                decodeProfile.attentionContext += System.nanoTime() - tc;
            }
        }

        t = prof ? System.nanoTime() : 0L;
        layer.outputProjection.projectInto(scratchContext, scratchAttnOut);
        if (prof) {
            decodeProfile.outputProjection += System.nanoTime() - t;
        }
    }

    private void runMlpStepInto(int layerIndex, boolean prof) {
        // GPU-resident MLP: gate_up → WARP SwiGLU → down in ONE submission, readback only after down.
        // (No CPU readback of gate/up and no re-upload of the intermediate between the two GEMMs.)
        long t = prof ? System.nanoTime() : 0L;
        mlpBlocks.get(layerIndex).project(scratchMlpNorm, scratchDown);
        if (prof) {
            decodeProfile.mlpPipeline += System.nanoTime() - t;
        }
    }

    // ── Batched prefill ───────────────────────────────────────────────────
    // Processes a block of >1 new tokens sequence-wise. Every dense projection runs ONCE per layer over the whole
    // block via projectSequenceInto(...) (one batched WARP dispatch instead of one per token). The CPU-side math
    // (RMSNorm, RoPE, causal grouped-query attention, SwiGLU, residuals) is identical to the per-token path, so the
    // batched prefill is numerically equivalent to the reference forward pass apart from GPU rounding.

    private float[] prefillSequence(List<Integer> tokenIds, DecoderOnlyWarpKvCache kvCache, int startPosition) {
        int hidden = config.hiddenSize();
        int seq = tokenIds.size() - startPosition;
        if (kvCache.layerCount() != layers.size()) {
            throw new IllegalArgumentException("KV cache layer count mismatch: expected "
                    + layers.size() + " but got " + kvCache.layerCount());
        }
        kvCache.requireReadyForPosition(startPosition);
        int lastPosition = tokenIds.size() - 1;
        if (lastPosition >= config.maxPositionEmbeddings()) {
            throw new IllegalArgumentException("position exceeds max_position_embeddings: " + lastPosition);
        }

        float[] hiddenSeq = new float[seq * hidden];
        for (int j = 0; j < seq; j++) {
            int tokenId = tokenIds.get(startPosition + j);
            if (tokenId < 0 || tokenId >= config.vocabSize()) {
                throw new IllegalArgumentException("tokenId out of vocabulary range: " + tokenId);
            }
            tokenEmbedding.copyRowInto(tokenId, scratchAttnNorm);
            System.arraycopy(scratchAttnNorm, 0, hiddenSeq, j * hidden, hidden);
        }
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            hiddenSeq = runLayerPrefill(hiddenSeq, seq, startPosition, layers.get(layerIndex),
                    kvCache.layer(layerIndex));
        }
        System.arraycopy(hiddenSeq, (seq - 1) * hidden, scratchFinalNorm, 0, hidden);
        return projectLogits(scratchFinalNorm);
    }

    private float[] runLayerPrefill(float[] hiddenSeq, int seq, int startPosition, DecoderOnlyWarpLayer layer,
                                    DecoderOnlyWarpLayerKvCache layerCache) {
        int hidden = config.hiddenSize();
        int numHeads = config.numAttentionHeads();
        int numKvHeads = config.numKeyValueHeads();
        int headDim = config.headDim();
        int qWidth = numHeads * headDim;
        int kvWidth = numKvHeads * headDim;

        // 1. attention RMSNorm per row (on a copy, identical to the reference; no per-row temp).
        float[] normSeq = hiddenSeq.clone();
        for (int j = 0; j < seq; j++) {
            DecoderOnlyMath.rmsNorm(normSeq, j * hidden, hidden, layer.inputNorm, rmsNormEps);
        }

        // 2. Q/K/V projection, fused and batched over the whole block (one WARP dispatch).
        float[] querySeq = new float[seq * qWidth];
        float[] keySeq = new float[seq * kvWidth];
        float[] valueSeq = new float[seq * kvWidth];
        float[] qkvSeq = new float[seq * layer.qkvProjection.totalOutputSize()];
        layer.qkvProjection.projectSequenceInto(normSeq, seq, qkvSeq);
        layer.qkvProjection.copySliceSequence(qkvSeq, seq, DecoderOnlyWarpLayer.QKV_QUERY, querySeq);
        layer.qkvProjection.copySliceSequence(qkvSeq, seq, DecoderOnlyWarpLayer.QKV_KEY, keySeq);
        layer.qkvProjection.copySliceSequence(qkvSeq, seq, DecoderOnlyWarpLayer.QKV_VALUE, valueSeq);

        // 3. RoPE per position, then append all K/V to the contiguous cache (in order, by offset).
        for (int j = 0; j < seq; j++) {
            int position = startPosition + j;
            for (int head = 0; head < numHeads; head++) {
                rotaryEmbedding.apply(querySeq, j * qWidth + head * headDim, position);
            }
            for (int head = 0; head < numKvHeads; head++) {
                rotaryEmbedding.apply(keySeq, j * kvWidth + head * headDim, position);
            }
        }
        for (int j = 0; j < seq; j++) {
            layerCache.append(keySeq, j * kvWidth, valueSeq, j * kvWidth);
        }

        // 4. Causal grouped-query attention per position (CPU), context packed [seq * hidden].
        float[] keys = layerCache.keys();
        float[] values = layerCache.values();
        int keyWidth = layerCache.keyWidth();
        float[] contextSeq = new float[seq * hidden];
        for (int j = 0; j < seq; j++) {
            int position = startPosition + j;
            int sourceCount = position + 1; // attend to sources 0..position (inclusive)
            for (int queryHead = 0; queryHead < numHeads; queryHead++) {
                int kvHead = attentionLayout.kvHeadForQueryHead(queryHead);
                int kvHeadOffset = kvHead * headDim;
                float[] scores = new float[sourceCount];
                for (int sourcePosition = 0; sourcePosition < sourceCount; sourcePosition++) {
                    scores[sourcePosition] = dotOffset(
                            querySeq, j * qWidth + queryHead * headDim,
                            keys, sourcePosition * keyWidth + kvHeadOffset,
                            headDim) * attentionScale;
                }
                DecoderOnlyMath.softmax(scores, scores.length);
                int contextOffset = j * hidden + queryHead * headDim;
                for (int sourcePosition = 0; sourcePosition < sourceCount; sourcePosition++) {
                    addWeightedOffset(contextSeq, contextOffset,
                            values, sourcePosition * keyWidth + kvHeadOffset, headDim, scores[sourcePosition]);
                }
            }
        }

        // 5. Output projection, batched.
        float[] attentionOut = new float[seq * hidden];
        layer.outputProjection.projectSequenceInto(contextSeq, seq, attentionOut);

        // 6. Residual.
        float[] afterAttention = new float[seq * hidden];
        for (int i = 0; i < afterAttention.length; i++) {
            afterAttention[i] = hiddenSeq[i] + attentionOut[i];
        }

        // 7. MLP RMSNorm per row (no per-row temp).
        float[] mlpNorm = afterAttention.clone();
        for (int j = 0; j < seq; j++) {
            DecoderOnlyMath.rmsNorm(mlpNorm, j * hidden, hidden, layer.postAttentionNorm, rmsNormEps);
        }

        // 8. Gate/Up projection, fused and batched (one WARP dispatch).
        int intermediate = layer.gateUpProjection.sliceSize(DecoderOnlyWarpLayer.GATE_UP_GATE);
        float[] gateSeq = new float[seq * intermediate];
        float[] upSeq = new float[seq * intermediate];
        float[] gateUpSeq = new float[seq * layer.gateUpProjection.totalOutputSize()];
        layer.gateUpProjection.projectSequenceInto(mlpNorm, seq, gateUpSeq);
        layer.gateUpProjection.copySliceSequence(gateUpSeq, seq, DecoderOnlyWarpLayer.GATE_UP_GATE, gateSeq);
        layer.gateUpProjection.copySliceSequence(gateUpSeq, seq, DecoderOnlyWarpLayer.GATE_UP_UP, upSeq);

        // 9. SwiGLU (element-wise, so applying over the whole packed buffer equals per-row).
        DecoderOnlyReferenceDenseOps.gatedSiluMultiply(gateSeq, upSeq);

        // 10. Down projection, batched.
        float[] downSeq = new float[seq * hidden];
        layer.downProjection.projectSequenceInto(gateSeq, seq, downSeq);

        // 11. Residual.
        float[] result = new float[seq * hidden];
        for (int i = 0; i < result.length; i++) {
            result[i] = afterAttention[i] + downSeq[i];
        }
        return result;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Decoder-only WARP forward pass is closed");
        }
    }

    /** Whether {@link #close()} has been called. */
    public boolean isClosed() {
        return closed;
    }

    /**
     * Idempotent close of every native/GPU resource this pass owns after a successful constructor:
     * the per-layer projections, the WARP LM head, the shared WARP SwiGLU kernel and the GPU-resident MLP pipeline.
     *
     * <p>Ownership note: {@code mlpBlocks} hold no own native resources — their barriers live in the
     * {@link #mlpPipeline}'s arena (freed by {@code mlpPipeline.close()}) and their gate-up/down kernels are owned by
     * the {@link #layers} (freed by {@code layer.close()}), so they need no separate close. {@code lmHeadReference} is
     * a CPU-only reference projection with no native resources.</p>
     *
     * <p>Every resource is closed even if an earlier close throws; the first failure is rethrown with the rest
     * attached as suppressed exceptions, so nothing leaks after the first error.</p>
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        // Reverse build order: SwiGLU + MLP pipeline first (built last), then layers, then the LM head.
        failure = closeResource(failure, swiGluKernel);
        failure = closeResource(failure, mlpPipeline);
        for (DecoderOnlyWarpLayer layer : layers) {
            failure = closeResource(failure, layer);
        }
        failure = closeResource(failure, lmHead); // null when the dev reference LM head is in use
        if (failure != null) {
            throw failure;
        }
    }

    private static RuntimeException closeResource(RuntimeException failure, AutoCloseable resource) {
        if (resource == null) {
            return failure;
        }
        try {
            resource.close();
        } catch (Exception e) {
            if (failure == null) {
                return new RuntimeException("Decoder-only WARP forward pass close() failed", e);
            }
            failure.addSuppressed(e);
        }
        return failure;
    }

    private static float dotOffset(float[] left, int leftOffset, float[] right, int rightOffset, int length) {
        float sum = 0.0f;
        for (int i = 0; i < length; i++) {
            sum += left[leftOffset + i] * right[rightOffset + i];
        }
        return sum;
    }

    private static void addWeightedOffset(float[] target, int targetOffset,
                                          float[] source, int sourceOffset, int length, float weight) {
        for (int i = 0; i < length; i++) {
            target[targetOffset + i] += weight * source[sourceOffset + i];
        }
    }
}
