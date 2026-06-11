package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyAttentionLayout;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyReferenceDenseOps;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyRotaryEmbedding;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyWarpDenseProjection;
import com.aresstack.windirectml.windows.WindowsBindings;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Native DirectML/WARP forward pass for the SmolLM2 decoder-only family.
 *
 * <p>This is the executable counterpart of {@link SmolLM2ReferenceForwardPass}. It keeps the identical numerical
 * structure (RMSNorm, RoPE, grouped-query attention, SwiGLU MLP, tied/untied LM head) but runs every dense
 * projection on the shared {@link DecoderOnlyWarpDenseProjection} GPU kernel instead of the CPU reference matmul.
 * Norms, RoPE, attention scoring and the SwiGLU non-linearity reuse the shared {@code decoderonly} math so the WARP
 * path stays byte-for-byte aligned with the reference path apart from GPU floating-point rounding.</p>
 *
 * <p>The projections are uploaded once at construction and reused across every decode step. Attention and the
 * KV cache run on the CPU exactly like the reference path; only the matmul-heavy projections move to WARP.</p>
 */
final class SmolLM2WarpForwardPass implements AutoCloseable {

    private final SmolLM2Config config;
    private final DecoderOnlyAttentionLayout attentionLayout;
    private final DecoderOnlyRotaryEmbedding rotaryEmbedding;
    private final float attentionScale;
    private final float rmsNormEps;

    private final SmolLM2DenseTensor tokenEmbedding;
    private final float[] finalNorm;
    private final List<WarpLayer> layers;
    private final DecoderOnlyWarpDenseProjection lmHead;

    /** LM-head projection time accumulated during the most recent {@link #logitsForLastToken} call. */
    private long lastCallLmHeadNanos;

    private boolean closed;

    SmolLM2WarpForwardPass(WindowsBindings windowsBindings, SmolLM2Weights weights) {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        SmolLM2ReferenceWeights referenceWeights = SmolLM2ReferenceWeights.from(weights);
        this.config = referenceWeights.config();
        this.attentionLayout = new DecoderOnlyAttentionLayout(
                config.numAttentionHeads(), config.effectiveKeyValueHeads());
        this.rotaryEmbedding = new DecoderOnlyRotaryEmbedding(
                config.effectiveHeadDim(), config.ropeTheta(), config.maxPositionEmbeddings());
        this.attentionScale = (float) (1.0d / Math.sqrt(config.effectiveHeadDim()));
        this.rmsNormEps = (float) config.rmsNormEps();
        this.tokenEmbedding = referenceWeights.tokenEmbedding();
        this.finalNorm = referenceWeights.finalNorm().copyValues();

        List<WarpLayer> builtLayers = new ArrayList<>(referenceWeights.layers().size());
        try {
            for (SmolLM2ReferenceLayerWeights layer : referenceWeights.layers()) {
                builtLayers.add(WarpLayer.build(windowsBindings, layer));
            }
            SmolLM2DenseTensor lmHeadTensor = referenceWeights.lmHeadTiedToEmbedding()
                    ? referenceWeights.tokenEmbedding()
                    : referenceWeights.lmHead();
            this.lmHead = projection(windowsBindings, "lm_head", lmHeadTensor);
        } catch (RuntimeException e) {
            for (WarpLayer layer : builtLayers) {
                layer.close();
            }
            throw e;
        }
        this.layers = List.copyOf(builtLayers);
    }

    SmolLM2Config config() {
        return config;
    }

    /**
     * LM-head projection time (nanoseconds) measured during the most recent {@link #logitsForLastToken} call. The
     * generation loop reports this separately so the LM-head cost is not silently folded into prefill/decoder timings.
     */
    long lastCallLmHeadNanos() {
        return lastCallLmHeadNanos;
    }

    /**
     * Decode the supplied token sequence with an incremental KV cache and return the logits for the last token.
     * Mirrors {@link SmolLM2ReferenceForwardPass#logitsForLastToken(List, SmolLM2ReferenceKvCache)}.
     *
     * <p>When more than one new token must be processed (the initial prompt prefill) the whole new-token block is
     * processed sequence-wise: every dense projection runs once per layer over the full block through
     * {@link DecoderOnlyWarpDenseProjection#projectSequenceInto}, instead of one WARP dispatch per token per
     * projection. This is mathematically identical to the per-token loop (same RMSNorm/RoPE/causal-attention/SwiGLU)
     * but issues far fewer WARP dispatches. The single-token decode step keeps the per-token path.</p>
     */
    float[] logitsForLastToken(List<Integer> tokenIds, SmolLM2ReferenceKvCache kvCache) {
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

    private float[] logitsForToken(int tokenId, int position, SmolLM2ReferenceKvCache kvCache, boolean projectLogits) {
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
        float[] hidden = tokenEmbedding.copyRow(tokenId);
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            hidden = runLayerStep(hidden, layers.get(layerIndex), kvCache.layer(layerIndex), position);
        }
        if (!projectLogits) {
            return null;
        }
        return projectLogits(hidden);
    }

    /** Apply the final RMSNorm and project the LM head, accumulating the LM-head time separately. */
    private float[] projectLogits(float[] hidden) {
        float[] normalized = hidden.clone();
        DecoderOnlyMath.rmsNorm(normalized, finalNorm, rmsNormEps);
        long lmHeadStart = System.nanoTime();
        float[] logits = lmHead.project(normalized);
        lastCallLmHeadNanos += System.nanoTime() - lmHeadStart;
        return logits;
    }

    private float[] runLayerStep(float[] hidden, WarpLayer layer,
                                 SmolLM2ReferenceLayerKvCache layerCache, int position) {
        float[] attentionInput = hidden.clone();
        DecoderOnlyMath.rmsNorm(attentionInput, layer.inputNorm, rmsNormEps);
        float[] attentionOutput = runSelfAttentionStep(attentionInput, layer, layerCache, position);
        float[] afterAttention = add(hidden, attentionOutput);

        float[] mlpInput = afterAttention.clone();
        DecoderOnlyMath.rmsNorm(mlpInput, layer.postAttentionNorm, rmsNormEps);
        float[] mlpOutput = runMlpStep(mlpInput, layer);
        return add(afterAttention, mlpOutput);
    }

    private float[] runSelfAttentionStep(float[] input, WarpLayer layer,
                                         SmolLM2ReferenceLayerKvCache layerCache, int position) {
        int hiddenSize = config.hiddenSize();
        int numHeads = config.numAttentionHeads();
        int numKvHeads = config.effectiveKeyValueHeads();
        int headDim = config.effectiveHeadDim();

        float[] query = layer.queryProjection.project(input);
        float[] key = layer.keyProjection.project(input);
        float[] value = layer.valueProjection.project(input);
        for (int head = 0; head < numHeads; head++) {
            rotaryEmbedding.apply(query, head * headDim, position);
        }
        for (int head = 0; head < numKvHeads; head++) {
            rotaryEmbedding.apply(key, head * headDim, position);
        }
        layerCache.append(key, value);

        float[] context = new float[hiddenSize];
        int sourceCount = layerCache.size();
        for (int queryHead = 0; queryHead < numHeads; queryHead++) {
            int kvHead = attentionLayout.kvHeadForQueryHead(queryHead);
            float[] scores = new float[sourceCount];
            for (int sourcePosition = 0; sourcePosition < sourceCount; sourcePosition++) {
                scores[sourcePosition] = dotHead(
                        query, queryHead,
                        layerCache.keyAt(sourcePosition), kvHead,
                        headDim) * attentionScale;
            }
            DecoderOnlyMath.softmax(scores, scores.length);
            int contextOffset = queryHead * headDim;
            for (int sourcePosition = 0; sourcePosition < sourceCount; sourcePosition++) {
                addHeadWeighted(context, contextOffset, layerCache.valueAt(sourcePosition), kvHead, headDim,
                        scores[sourcePosition]);
            }
        }
        return layer.outputProjection.project(context);
    }

    private float[] runMlpStep(float[] input, WarpLayer layer) {
        float[] gate = layer.gateProjection.project(input);
        float[] up = layer.upProjection.project(input);
        if (gate.length != up.length) {
            throw new IllegalStateException("SmolLM2 gate and up projections must have the same width");
        }
        DecoderOnlyReferenceDenseOps.gatedSiluMultiply(gate, up);
        return layer.downProjection.project(gate);
    }

    // ── Batched prefill ───────────────────────────────────────────────────
    // Processes a block of >1 new tokens sequence-wise. Every dense projection runs ONCE per layer over the whole
    // block via projectSequenceInto(...) (one batched WARP dispatch instead of one per token). The CPU-side math
    // (RMSNorm, RoPE, causal grouped-query attention, SwiGLU, residuals) is identical to the per-token path, so the
    // batched prefill is numerically equivalent to the reference forward pass apart from GPU rounding.

    private float[] prefillSequence(List<Integer> tokenIds, SmolLM2ReferenceKvCache kvCache, int startPosition) {
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
            float[] row = tokenEmbedding.copyRow(tokenId);
            System.arraycopy(row, 0, hiddenSeq, j * hidden, hidden);
        }
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            hiddenSeq = runLayerPrefill(hiddenSeq, seq, startPosition, layers.get(layerIndex),
                    kvCache.layer(layerIndex));
        }
        float[] last = new float[hidden];
        System.arraycopy(hiddenSeq, (seq - 1) * hidden, last, 0, hidden);
        return projectLogits(last);
    }

    private float[] runLayerPrefill(float[] hiddenSeq, int seq, int startPosition, WarpLayer layer,
                                    SmolLM2ReferenceLayerKvCache layerCache) {
        int hidden = config.hiddenSize();
        int numHeads = config.numAttentionHeads();
        int numKvHeads = config.effectiveKeyValueHeads();
        int headDim = config.effectiveHeadDim();
        int qSize = numHeads * headDim;
        int kvSize = numKvHeads * headDim;

        // 1. attention RMSNorm per row (on a copy, identical to the reference).
        float[] normSeq = hiddenSeq.clone();
        for (int j = 0; j < seq; j++) {
            rmsNormRow(normSeq, j * hidden, layer.inputNorm, hidden);
        }

        // 2. Q/K/V projections, batched over the whole block.
        float[] querySeq = new float[seq * qSize];
        float[] keySeq = new float[seq * kvSize];
        float[] valueSeq = new float[seq * kvSize];
        layer.queryProjection.projectSequenceInto(normSeq, seq, querySeq);
        layer.keyProjection.projectSequenceInto(normSeq, seq, keySeq);
        layer.valueProjection.projectSequenceInto(normSeq, seq, valueSeq);

        // 3. RoPE per position, then append all K/V to the cache (in order).
        for (int j = 0; j < seq; j++) {
            int position = startPosition + j;
            for (int head = 0; head < numHeads; head++) {
                rotaryEmbedding.apply(querySeq, j * qSize + head * headDim, position);
            }
            for (int head = 0; head < numKvHeads; head++) {
                rotaryEmbedding.apply(keySeq, j * kvSize + head * headDim, position);
            }
        }
        for (int j = 0; j < seq; j++) {
            float[] key = Arrays.copyOfRange(keySeq, j * kvSize, (j + 1) * kvSize);
            float[] value = Arrays.copyOfRange(valueSeq, j * kvSize, (j + 1) * kvSize);
            layerCache.append(key, value);
        }

        // 4. Causal grouped-query attention per position (CPU), context packed [seq * hidden].
        float[] contextSeq = new float[seq * hidden];
        for (int j = 0; j < seq; j++) {
            int position = startPosition + j;
            int sourceCount = position + 1; // attend to sources 0..position (inclusive)
            for (int queryHead = 0; queryHead < numHeads; queryHead++) {
                int kvHead = attentionLayout.kvHeadForQueryHead(queryHead);
                float[] scores = new float[sourceCount];
                for (int sourcePosition = 0; sourcePosition < sourceCount; sourcePosition++) {
                    scores[sourcePosition] = dotHeadSeq(
                            querySeq, j * qSize, queryHead,
                            layerCache.keyAt(sourcePosition), kvHead,
                            headDim) * attentionScale;
                }
                DecoderOnlyMath.softmax(scores, scores.length);
                int contextOffset = j * hidden + queryHead * headDim;
                for (int sourcePosition = 0; sourcePosition < sourceCount; sourcePosition++) {
                    addHeadWeighted(contextSeq, contextOffset, layerCache.valueAt(sourcePosition), kvHead, headDim,
                            scores[sourcePosition]);
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

        // 7. MLP RMSNorm per row.
        float[] mlpNorm = afterAttention.clone();
        for (int j = 0; j < seq; j++) {
            rmsNormRow(mlpNorm, j * hidden, layer.postAttentionNorm, hidden);
        }

        // 8. Gate/Up projections, batched.
        int intermediate = layer.gateProjection.outputSize();
        if (layer.upProjection.outputSize() != intermediate) {
            throw new IllegalStateException("SmolLM2 gate and up projections must have the same width");
        }
        float[] gateSeq = new float[seq * intermediate];
        float[] upSeq = new float[seq * intermediate];
        layer.gateProjection.projectSequenceInto(mlpNorm, seq, gateSeq);
        layer.upProjection.projectSequenceInto(mlpNorm, seq, upSeq);

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

    private void rmsNormRow(float[] sequence, int offset, float[] weight, int hidden) {
        float[] row = new float[hidden];
        System.arraycopy(sequence, offset, row, 0, hidden);
        DecoderOnlyMath.rmsNorm(row, weight, rmsNormEps);
        System.arraycopy(row, 0, sequence, offset, hidden);
    }

    private static float dotHeadSeq(float[] left, int leftBase, int leftHead,
                                    float[] right, int rightHead, int headDim) {
        int leftOffset = leftBase + leftHead * headDim;
        int rightOffset = rightHead * headDim;
        float sum = 0.0f;
        for (int i = 0; i < headDim; i++) {
            sum += left[leftOffset + i] * right[rightOffset + i];
        }
        return sum;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("SmolLM2 WARP forward pass is closed");
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        for (WarpLayer layer : layers) {
            layer.close();
        }
        lmHead.close();
    }

    private static float dotHead(float[] left, int leftHead, float[] right, int rightHead, int headDim) {
        int leftOffset = leftHead * headDim;
        int rightOffset = rightHead * headDim;
        float sum = 0.0f;
        for (int i = 0; i < headDim; i++) {
            sum += left[leftOffset + i] * right[rightOffset + i];
        }
        return sum;
    }

    private static void addHeadWeighted(float[] target, int targetOffset,
                                        float[] source, int sourceHead, int headDim, float weight) {
        int sourceOffset = sourceHead * headDim;
        for (int i = 0; i < headDim; i++) {
            target[targetOffset + i] += weight * source[sourceOffset + i];
        }
    }

    private static float[] add(float[] left, float[] right) {
        if (left.length != right.length) {
            throw new IllegalArgumentException("hidden size mismatch");
        }
        float[] result = new float[left.length];
        for (int i = 0; i < left.length; i++) {
            result[i] = left[i] + right[i];
        }
        return result;
    }

    private static DecoderOnlyWarpDenseProjection projection(WindowsBindings windowsBindings,
                                                             String name,
                                                             SmolLM2DenseTensor tensor) {
        if (tensor.rank() != 2) {
            throw new IllegalStateException("SmolLM2 WARP projection '" + name + "' must be rank-2 but was rank "
                    + tensor.rank());
        }
        int outputSize = tensor.dim(0);
        int inputSize = tensor.dim(1);
        return DecoderOnlyWarpDenseProjection.fromRowMajorWeights(
                windowsBindings, name, outputSize, inputSize, tensor.copyValues());
    }

    /**
     * GPU-resident projections + CPU-resident norm vectors for a single decoder layer.
     */
    private static final class WarpLayer implements AutoCloseable {
        private final float[] inputNorm;
        private final float[] postAttentionNorm;
        private final DecoderOnlyWarpDenseProjection queryProjection;
        private final DecoderOnlyWarpDenseProjection keyProjection;
        private final DecoderOnlyWarpDenseProjection valueProjection;
        private final DecoderOnlyWarpDenseProjection outputProjection;
        private final DecoderOnlyWarpDenseProjection gateProjection;
        private final DecoderOnlyWarpDenseProjection upProjection;
        private final DecoderOnlyWarpDenseProjection downProjection;

        private WarpLayer(float[] inputNorm,
                          float[] postAttentionNorm,
                          DecoderOnlyWarpDenseProjection queryProjection,
                          DecoderOnlyWarpDenseProjection keyProjection,
                          DecoderOnlyWarpDenseProjection valueProjection,
                          DecoderOnlyWarpDenseProjection outputProjection,
                          DecoderOnlyWarpDenseProjection gateProjection,
                          DecoderOnlyWarpDenseProjection upProjection,
                          DecoderOnlyWarpDenseProjection downProjection) {
            this.inputNorm = inputNorm;
            this.postAttentionNorm = postAttentionNorm;
            this.queryProjection = queryProjection;
            this.keyProjection = keyProjection;
            this.valueProjection = valueProjection;
            this.outputProjection = outputProjection;
            this.gateProjection = gateProjection;
            this.upProjection = upProjection;
            this.downProjection = downProjection;
        }

        static WarpLayer build(WindowsBindings windowsBindings, SmolLM2ReferenceLayerWeights layer) {
            int index = layer.layerIndex();
            List<DecoderOnlyWarpDenseProjection> built = new ArrayList<>(7);
            try {
                DecoderOnlyWarpDenseProjection q = projection(windowsBindings, "layer." + index + ".q_proj", layer.queryProjection());
                built.add(q);
                DecoderOnlyWarpDenseProjection k = projection(windowsBindings, "layer." + index + ".k_proj", layer.keyProjection());
                built.add(k);
                DecoderOnlyWarpDenseProjection v = projection(windowsBindings, "layer." + index + ".v_proj", layer.valueProjection());
                built.add(v);
                DecoderOnlyWarpDenseProjection o = projection(windowsBindings, "layer." + index + ".o_proj", layer.outputProjection());
                built.add(o);
                DecoderOnlyWarpDenseProjection gate = projection(windowsBindings, "layer." + index + ".gate_proj", layer.gateProjection());
                built.add(gate);
                DecoderOnlyWarpDenseProjection up = projection(windowsBindings, "layer." + index + ".up_proj", layer.upProjection());
                built.add(up);
                DecoderOnlyWarpDenseProjection down = projection(windowsBindings, "layer." + index + ".down_proj", layer.downProjection());
                built.add(down);
                return new WarpLayer(
                        layer.inputNorm().copyValues(),
                        layer.postAttentionNorm().copyValues(),
                        q, k, v, o, gate, up, down);
            } catch (RuntimeException e) {
                for (DecoderOnlyWarpDenseProjection projection : built) {
                    projection.close();
                }
                throw e;
            }
        }

        @Override
        public void close() {
            queryProjection.close();
            keyProjection.close();
            valueProjection.close();
            outputProjection.close();
            gateProjection.close();
            upProjection.close();
            downProjection.close();
        }
    }
}
