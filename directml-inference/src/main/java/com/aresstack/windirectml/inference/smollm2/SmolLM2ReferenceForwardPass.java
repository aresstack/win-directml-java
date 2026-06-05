package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyAttentionLayout;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyRotaryEmbedding;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Correctness-first SmolLM2 forward pass for tiny tests and runtime bring-up.
 *
 * <p>The implementation supports both full-sequence validation and incremental decoding with a reference KV cache.
 * It exists to validate roles, tensor layout and generation semantics before optimized kernels are introduced.</p>
 */
public final class SmolLM2ReferenceForwardPass {

    private final SmolLM2ReferenceWeights weights;
    private final SmolLM2Config config;
    private final DecoderOnlyAttentionLayout attentionLayout;
    private final DecoderOnlyRotaryEmbedding rotaryEmbedding;
    private final float attentionScale;
    private final SmolLM2ReferenceForwardProfile profile;

    public SmolLM2ReferenceForwardPass(SmolLM2Weights weights) {
        this(SmolLM2ReferenceWeights.from(weights));
    }

    public SmolLM2ReferenceForwardPass(SmolLM2Weights weights, SmolLM2ReferenceForwardProfile profile) {
        this(SmolLM2ReferenceWeights.from(weights), profile);
    }

    public SmolLM2ReferenceForwardPass(SmolLM2ReferenceWeights weights) {
        this(weights, new SmolLM2ReferenceForwardProfile());
    }

    SmolLM2ReferenceForwardPass(SmolLM2ReferenceWeights weights, SmolLM2ReferenceForwardProfile profile) {
        this.weights = Objects.requireNonNull(weights, "weights");
        this.config = weights.config();
        this.profile = Objects.requireNonNull(profile, "profile");
        this.attentionLayout = new DecoderOnlyAttentionLayout(
                config.numAttentionHeads(), config.effectiveKeyValueHeads());
        this.rotaryEmbedding = new DecoderOnlyRotaryEmbedding(
                config.effectiveHeadDim(), config.ropeTheta(), config.maxPositionEmbeddings());
        this.attentionScale = (float) (1.0d / Math.sqrt(config.effectiveHeadDim()));
    }

    public SmolLM2Config config() {
        return config;
    }

    SmolLM2ReferenceForwardProfile profile() {
        return profile;
    }

    public float[] logitsForLastToken(List<Integer> tokenIds) {
        Objects.requireNonNull(tokenIds, "tokenIds");
        if (tokenIds.isEmpty()) {
            throw new IllegalArgumentException("tokenIds must not be empty");
        }
        float[][] hiddenStates = embedTokens(tokenIds);
        for (SmolLM2ReferenceLayerWeights layer : weights.layers()) {
            hiddenStates = runLayer(hiddenStates, layer);
        }
        float[] lastHidden = hiddenStates[hiddenStates.length - 1].clone();
        DecoderOnlyMath.rmsNorm(lastHidden, weights.finalNorm().copyValues(), (float) config.rmsNormEps());
        return projectToLogits(lastHidden);
    }

    public float[] logitsForLastToken(List<Integer> tokenIds, SmolLM2ReferenceKvCache kvCache) {
        Objects.requireNonNull(tokenIds, "tokenIds");
        Objects.requireNonNull(kvCache, "kvCache");
        if (tokenIds.isEmpty()) {
            throw new IllegalArgumentException("tokenIds must not be empty");
        }
        int startPosition = kvCache.completedTokenCount();
        if (startPosition > tokenIds.size()) {
            throw new IllegalArgumentException("KV cache contains more tokens than the supplied sequence");
        }
        float[] logits = null;
        for (int position = startPosition; position < tokenIds.size(); position++) {
            logits = logitsForToken(tokenIds.get(position), position, kvCache);
        }
        if (logits == null) {
            logits = logitsForLastToken(tokenIds);
        }
        return logits;
    }

    private float[] logitsForToken(int tokenId, int position, SmolLM2ReferenceKvCache kvCache) {
        if (tokenId < 0 || tokenId >= config.vocabSize()) {
            throw new IllegalArgumentException("tokenId out of vocabulary range: " + tokenId);
        }
        if (position >= config.maxPositionEmbeddings()) {
            throw new IllegalArgumentException("position exceeds max_position_embeddings: " + position);
        }
        kvCache.requireReadyForPosition(position);
        float[] hidden = weights.tokenEmbedding().copyRow(tokenId);
        List<SmolLM2ReferenceLayerWeights> layers = weights.layers();
        if (kvCache.layerCount() != layers.size()) {
            throw new IllegalArgumentException("KV cache layer count mismatch: expected "
                    + layers.size() + " but got " + kvCache.layerCount());
        }
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            hidden = runLayerStep(hidden, layers.get(layerIndex), kvCache.layer(layerIndex), position);
        }
        float[] normalized = hidden.clone();
        long finalNormStart = System.nanoTime();
        DecoderOnlyMath.rmsNorm(normalized, weights.finalNorm().copyValues(), (float) config.rmsNormEps());
        profile.addFinalNormNanos(System.nanoTime() - finalNormStart);
        return projectToLogits(normalized);
    }

    private float[][] embedTokens(List<Integer> tokenIds) {
        int hiddenSize = config.hiddenSize();
        float[][] hiddenStates = new float[tokenIds.size()][hiddenSize];
        for (int position = 0; position < tokenIds.size(); position++) {
            int tokenId = tokenIds.get(position);
            if (tokenId < 0 || tokenId >= config.vocabSize()) {
                throw new IllegalArgumentException("tokenId out of vocabulary range: " + tokenId);
            }
            hiddenStates[position] = weights.tokenEmbedding().copyRow(tokenId);
        }
        return hiddenStates;
    }

    private float[][] runLayer(float[][] hiddenStates, SmolLM2ReferenceLayerWeights layer) {
        float[][] attentionInput = copy2d(hiddenStates);
        for (float[] row : attentionInput) {
            DecoderOnlyMath.rmsNorm(row, layer.inputNorm().copyValues(), (float) config.rmsNormEps());
        }

        float[][] attentionOutput = runSelfAttention(attentionInput, layer);
        float[][] afterAttention = add(hiddenStates, attentionOutput);

        float[][] mlpInput = copy2d(afterAttention);
        for (float[] row : mlpInput) {
            DecoderOnlyMath.rmsNorm(row, layer.postAttentionNorm().copyValues(), (float) config.rmsNormEps());
        }
        float[][] mlpOutput = runMlp(mlpInput, layer);
        return add(afterAttention, mlpOutput);
    }

    private float[] runLayerStep(float[] hidden, SmolLM2ReferenceLayerWeights layer,
                                 SmolLM2ReferenceLayerKvCache layerCache, int position) {
        float[] attentionInput = hidden.clone();
        long attentionNormStart = System.nanoTime();
        DecoderOnlyMath.rmsNorm(attentionInput, layer.inputNorm().copyValues(), (float) config.rmsNormEps());
        profile.addLayerNormNanos(System.nanoTime() - attentionNormStart);
        float[] attentionOutput = runSelfAttentionStep(attentionInput, layer, layerCache, position);
        float[] afterAttention = add(hidden, attentionOutput);

        float[] mlpInput = afterAttention.clone();
        long mlpNormStart = System.nanoTime();
        DecoderOnlyMath.rmsNorm(mlpInput, layer.postAttentionNorm().copyValues(), (float) config.rmsNormEps());
        profile.addLayerNormNanos(System.nanoTime() - mlpNormStart);
        float[] mlpOutput = runMlpStep(mlpInput, layer);
        return add(afterAttention, mlpOutput);
    }

    private float[][] runSelfAttention(float[][] input, SmolLM2ReferenceLayerWeights layer) {
        int sequenceLength = input.length;
        int hiddenSize = config.hiddenSize();
        int numHeads = config.numAttentionHeads();
        int numKvHeads = config.effectiveKeyValueHeads();
        int headDim = config.effectiveHeadDim();

        float[][] queries = new float[sequenceLength][];
        float[][] keys = new float[sequenceLength][];
        float[][] values = new float[sequenceLength][];
        for (int position = 0; position < sequenceLength; position++) {
            queries[position] = layer.queryProjection().multiplyVector(input[position]);
            keys[position] = layer.keyProjection().multiplyVector(input[position]);
            values[position] = layer.valueProjection().multiplyVector(input[position]);
            for (int head = 0; head < numHeads; head++) {
                rotaryEmbedding.apply(queries[position], head * headDim, position);
            }
            for (int head = 0; head < numKvHeads; head++) {
                rotaryEmbedding.apply(keys[position], head * headDim, position);
            }
        }

        float[][] output = new float[sequenceLength][hiddenSize];
        for (int position = 0; position < sequenceLength; position++) {
            float[] context = new float[hiddenSize];
            for (int queryHead = 0; queryHead < numHeads; queryHead++) {
                int kvHead = attentionLayout.kvHeadForQueryHead(queryHead);
                float[] scores = new float[position + 1];
                for (int sourcePosition = 0; sourcePosition <= position; sourcePosition++) {
                    scores[sourcePosition] = dotHead(
                            queries[position], queryHead,
                            keys[sourcePosition], kvHead,
                            headDim) * attentionScale;
                }
                DecoderOnlyMath.softmax(scores, scores.length);
                int contextOffset = queryHead * headDim;
                for (int sourcePosition = 0; sourcePosition <= position; sourcePosition++) {
                    addHeadWeighted(context, contextOffset, values[sourcePosition], kvHead, headDim, scores[sourcePosition]);
                }
            }
            output[position] = layer.outputProjection().multiplyVector(context);
        }
        return output;
    }


    private float[] runSelfAttentionStep(float[] input, SmolLM2ReferenceLayerWeights layer,
                                         SmolLM2ReferenceLayerKvCache layerCache, int position) {
        int hiddenSize = config.hiddenSize();
        int numHeads = config.numAttentionHeads();
        int numKvHeads = config.effectiveKeyValueHeads();
        int headDim = config.effectiveHeadDim();

        long projectionStart = System.nanoTime();
        float[] query = layer.queryProjection().multiplyVector(input);
        float[] key = layer.keyProjection().multiplyVector(input);
        float[] value = layer.valueProjection().multiplyVector(input);
        profile.addAttentionProjectionNanos(System.nanoTime() - projectionStart);
        for (int head = 0; head < numHeads; head++) {
            rotaryEmbedding.apply(query, head * headDim, position);
        }
        for (int head = 0; head < numKvHeads; head++) {
            rotaryEmbedding.apply(key, head * headDim, position);
        }
        layerCache.append(key, value);

        long attentionScoreStart = System.nanoTime();
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
        profile.addAttentionScoreNanos(System.nanoTime() - attentionScoreStart);
        long outputProjectionStart = System.nanoTime();
        float[] output = layer.outputProjection().multiplyVector(context);
        profile.addAttentionOutputProjectionNanos(System.nanoTime() - outputProjectionStart);
        return output;
    }

    private float[][] runMlp(float[][] input, SmolLM2ReferenceLayerWeights layer) {
        float[][] output = new float[input.length][config.hiddenSize()];
        for (int position = 0; position < input.length; position++) {
            output[position] = runMlpStep(input[position], layer);
        }
        return output;
    }

    private float[] runMlpStep(float[] input, SmolLM2ReferenceLayerWeights layer) {
        long mlpStart = System.nanoTime();
        float[] gate = layer.gateProjection().multiplyVector(input);
        float[] up = layer.upProjection().multiplyVector(input);
        if (gate.length != up.length) {
            throw new IllegalStateException("SmolLM2 gate and up projections must have the same width");
        }
        for (int i = 0; i < gate.length; i++) {
            gate[i] = DecoderOnlyMath.fastSilu(gate[i]) * up[i];
        }
        float[] output = layer.downProjection().multiplyVector(gate);
        profile.addMlpNanos(System.nanoTime() - mlpStart);
        return output;
    }

    private float[] projectToLogits(float[] hidden) {
        long start = System.nanoTime();
        int vocabSize = config.vocabSize();
        float[] logits = new float[vocabSize];
        SmolLM2DenseTensor lmHead = weights.lmHeadTiedToEmbedding() ? weights.tokenEmbedding() : weights.lmHead();
        for (int tokenId = 0; tokenId < vocabSize; tokenId++) {
            logits[tokenId] = lmHead.dotRow(tokenId, hidden);
        }
        profile.addLmHeadNanos(System.nanoTime() - start);
        return logits;
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

    private static float[][] add(float[][] left, float[][] right) {
        if (left.length != right.length) {
            throw new IllegalArgumentException("sequence length mismatch");
        }
        float[][] result = new float[left.length][];
        for (int row = 0; row < left.length; row++) {
            if (left[row].length != right[row].length) {
                throw new IllegalArgumentException("hidden size mismatch at row " + row);
            }
            result[row] = new float[left[row].length];
            for (int col = 0; col < left[row].length; col++) {
                result[row][col] = left[row][col] + right[row][col];
            }
        }
        return result;
    }

    private static float[][] copy2d(float[][] values) {
        float[][] copy = new float[values.length][];
        for (int i = 0; i < values.length; i++) {
            copy[i] = Arrays.copyOf(values[i], values[i].length);
        }
        return copy;
    }
}
