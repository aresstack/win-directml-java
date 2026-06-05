package com.aresstack.windirectml.inference.t5;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * CPU reference T5 decoder pipeline.
 *
 * <p>Use this implementation to validate decoder self-attention,
 * encoder-decoder cross-attention, and cache API boundaries before replacing
 * the math with WARP kernels.</p>
 */
public final class T5DecoderPipeline implements T5DecoderRunner, AutoCloseable {
    private final T5PackageMetadata metadata;
    private final T5Weights weights;
    private final T5TensorData sharedEmbedding;
    private final List<T5DecoderBlock> blocks;
    private final T5LayerNorm finalLayerNorm;
    private final T5LinearProjectionFactory projectionFactory;
    private T5CrossAttentionMemorySet crossAttentionMemorySet;

    private T5DecoderPipeline(T5PackageMetadata metadata,
                              T5Weights weights,
                              T5TensorData sharedEmbedding,
                              List<T5DecoderBlock> blocks,
                              T5LayerNorm finalLayerNorm,
                              T5LinearProjectionFactory projectionFactory) {
        this.metadata = Objects.requireNonNull(metadata, "metadata");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.sharedEmbedding = Objects.requireNonNull(sharedEmbedding, "sharedEmbedding");
        this.blocks = List.copyOf(blocks);
        this.finalLayerNorm = Objects.requireNonNull(finalLayerNorm, "finalLayerNorm");
        this.projectionFactory = Objects.requireNonNull(projectionFactory, "projectionFactory");
    }

    public static T5DecoderPipeline from(T5Weights weights) {
        Objects.requireNonNull(weights, "weights");
        return from(weights, T5ReferenceLinearProjectionFactory.INSTANCE);
    }

    public static T5DecoderPipeline from(T5Weights weights, T5LinearProjectionFactory projectionFactory) {
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(projectionFactory, "projectionFactory");
        T5PackageMetadata metadata = weights.metadata();
        T5TensorData sharedEmbedding = T5TensorData.from(weights.sharedEmbedding());
        T5TensorData decoderBias = optional(weights, "decoder.relative_attention_bias");
        T5RelativePositionBias relativePositionBias = new T5RelativePositionBias(
                decoderBias, metadata.relativeAttentionBuckets(), metadata.relativeAttentionMaxDistance(), metadata.numHeads());
        List<T5DecoderBlock> blocks = new ArrayList<>();
        for (int layer = 0; layer < metadata.decoderLayers(); layer++) {
            blocks.add(createBlock(weights, metadata, layer, relativePositionBias, projectionFactory));
        }
        T5LayerNorm finalLayerNorm = new T5LayerNorm(T5TensorData.from(weights.decoderFinalLayerNorm()), 1e-6f);
        return new T5DecoderPipeline(metadata, weights, sharedEmbedding, blocks, finalLayerNorm, projectionFactory);
    }

    @Override
    public String executionMode() {
        return "reference-decoder";
    }

    @Override
    public T5DecoderState decode(int[] decoderInputIds, T5EncoderOutput encoderOutput) {
        validateInput(decoderInputIds);
        Objects.requireNonNull(encoderOutput, "encoderOutput");
        boolean[] decoderMask = decoderMask(decoderInputIds.length);
        int sequenceLength = decoderInputIds.length;
        int hiddenSize = metadata.dModel();
        float[] hiddenStates = embed(decoderInputIds);
        T5CrossAttentionMemorySet memorySet = crossAttentionMemorySetFor(encoderOutput);
        for (int layer = 0; layer < blocks.size(); layer++) {
            hiddenStates = blocks.get(layer).apply(hiddenStates, sequenceLength, hiddenSize,
                    decoderMask, memorySet.memory(layer));
        }
        hiddenStates = finalLayerNorm.applySequence(hiddenStates, sequenceLength, hiddenSize);
        return new T5DecoderState(sequenceLength, hiddenSize, hiddenStates);
    }

    @Override
    public T5DecoderState decodeStep(int decoderTokenId, T5EncoderOutput encoderOutput, T5DecoderCache cache) {
        T5DecoderCache safeCache = cache == null ? T5DecoderCache.empty() : cache;
        if (!safeCache.isEmpty() && !safeCache.hasSelfAttentionMemories(blocks.size())) {
            return decode(safeCache.withAppendedToken(decoderTokenId), encoderOutput);
        }
        return decodeIncremental(decoderTokenId, encoderOutput, safeCache);
    }


    private T5DecoderState decodeIncremental(int decoderTokenId, T5EncoderOutput encoderOutput, T5DecoderCache cache) {
        Objects.requireNonNull(encoderOutput, "encoderOutput");
        validateToken(decoderTokenId);
        int tokenIndex = cache.generatedTokens();
        int hiddenSize = metadata.dModel();
        float[] hiddenState = embedToken(decoderTokenId);
        T5CrossAttentionMemorySet memorySet = crossAttentionMemorySetFor(encoderOutput);
        List<T5SelfAttentionMemory> nextLayerMemories = new ArrayList<>();
        for (int layer = 0; layer < blocks.size(); layer++) {
            T5DecoderBlockStep step = blocks.get(layer).applyStep(hiddenState, tokenIndex,
                    cache.selfAttentionMemory(layer), memorySet.memory(layer));
            hiddenState = step.hiddenStateUnsafe();
            nextLayerMemories.add(step.selfAttentionMemory());
        }
        float[] finalHiddenState = finalLayerNorm.applySequence(hiddenState, 1, hiddenSize);
        float[] nextHiddenStates = appendHiddenState(cache.hiddenStates(), finalHiddenState, hiddenSize);
        int[] nextTokenIds = cache.withAppendedToken(decoderTokenId);
        T5DecoderCache nextCache = T5DecoderCache.fromIncrementalState(nextTokenIds, hiddenSize,
                nextHiddenStates, nextLayerMemories);
        return T5DecoderState.withNextCache(nextTokenIds.length, hiddenSize, nextHiddenStates, nextCache);
    }

    private float[] embedToken(int tokenId) {
        validateToken(tokenId);
        int hiddenSize = metadata.dModel();
        float[] result = new float[hiddenSize];
        for (int dim = 0; dim < hiddenSize; dim++) {
            result[dim] = sharedEmbedding.at(tokenId, dim);
        }
        return result;
    }

    private static float[] appendHiddenState(float[] existing, float[] appended, int hiddenSize) {
        if (appended.length != hiddenSize) {
            throw new IllegalArgumentException("appended hidden state length mismatch: " + appended.length
                    + ", expected=" + hiddenSize);
        }
        float[] result = java.util.Arrays.copyOf(existing, existing.length + hiddenSize);
        System.arraycopy(appended, 0, result, existing.length, hiddenSize);
        return result;
    }

    private void validateToken(int tokenId) {
        if (tokenId < 0) {
            throw new IllegalArgumentException("decoderInputIds must not contain negative token ids");
        }
        if (tokenId >= sharedEmbedding.dim(0)) {
            throw new IllegalArgumentException("decoderInputIds contains token outside vocabulary: " + tokenId);
        }
    }

    public T5Weights weights() {
        return weights;
    }

    private T5CrossAttentionMemorySet crossAttentionMemorySetFor(T5EncoderOutput encoderOutput) {
        T5CrossAttentionMemorySet cached = crossAttentionMemorySet;
        if (cached != null && cached.belongsTo(encoderOutput)) {
            return cached;
        }
        long prepareStart = System.nanoTime();
        List<T5CrossAttentionMemory> memories = new ArrayList<>();
        for (T5DecoderBlock block : blocks) {
            memories.add(block.prepareCrossAttentionMemory(encoderOutput));
        }
        T5GenerationProfiler.recordCrossAttentionPrepareNanos(System.nanoTime() - prepareStart);
        T5CrossAttentionMemorySet created = new T5CrossAttentionMemorySet(encoderOutput, memories);
        crossAttentionMemorySet = created;
        return created;
    }

    private float[] embed(int[] decoderInputIds) {
        int sequenceLength = decoderInputIds.length;
        int hiddenSize = metadata.dModel();
        float[] result = new float[sequenceLength * hiddenSize];
        for (int token = 0; token < sequenceLength; token++) {
            int tokenId = decoderInputIds[token];
            if (tokenId >= sharedEmbedding.dim(0)) {
                throw new IllegalArgumentException("decoderInputIds contains token outside vocabulary: " + tokenId);
            }
            for (int dim = 0; dim < hiddenSize; dim++) {
                result[token * hiddenSize + dim] = sharedEmbedding.at(tokenId, dim);
            }
        }
        return result;
    }

    private static T5DecoderBlock createBlock(T5Weights weights,
                                              T5PackageMetadata metadata,
                                              int layer,
                                              T5RelativePositionBias relativePositionBias,
                                              T5LinearProjectionFactory projectionFactory) {
        T5LayerNorm selfAttentionLayerNorm = new T5LayerNorm(
                T5TensorData.from(weights.decoderLayerNorm(layer, 0)), 1e-6f);
        T5SelfAttention selfAttention = new T5SelfAttention(metadata,
                projectionFactory.createSelfAttentionProjection(
                        T5TensorData.from(weights.decoderSelfAttention(layer, "q")),
                        T5TensorData.from(weights.decoderSelfAttention(layer, "k")),
                        T5TensorData.from(weights.decoderSelfAttention(layer, "v"))),
                projectionFactory.create(T5TensorData.from(weights.decoderSelfAttention(layer, "o"))),
                relativePositionBias,
                false,
                true);
        T5LayerNorm crossAttentionLayerNorm = new T5LayerNorm(
                T5TensorData.from(weights.decoderLayerNorm(layer, 1)), 1e-6f);
        T5CrossAttention crossAttention = new T5CrossAttention(metadata,
                projectionFactory.create(T5TensorData.from(weights.decoderCrossAttention(layer, "q"))),
                projectionFactory.createCrossAttentionMemoryProjection(
                        T5TensorData.from(weights.decoderCrossAttention(layer, "k")),
                        T5TensorData.from(weights.decoderCrossAttention(layer, "v"))),
                projectionFactory.create(T5TensorData.from(weights.decoderCrossAttention(layer, "o"))));
        T5LayerNorm feedForwardLayerNorm = new T5LayerNorm(
                T5TensorData.from(weights.decoderLayerNorm(layer, 2)), 1e-6f);
        T5FeedForward feedForward = new T5FeedForward(metadata,
                optionalProjection(weights, "decoder.layer." + layer + ".ffn.wi", projectionFactory),
                optionalProjection(weights, "decoder.layer." + layer + ".ffn.wi_0", projectionFactory),
                optionalProjection(weights, "decoder.layer." + layer + ".ffn.wi_1", projectionFactory),
                projectionFactory.create(T5TensorData.from(weights.decoderFeedForward(layer, "wo"))));
        return new T5DecoderBlock(selfAttentionLayerNorm, selfAttention, crossAttentionLayerNorm,
                crossAttention, feedForwardLayerNorm, feedForward);
    }

    private static T5TensorData optional(T5Weights weights, String role) {
        if (weights.optional(role) == null) {
            return null;
        }
        return T5TensorData.from(weights.optional(role));
    }

    private static T5LinearProjection optionalProjection(T5Weights weights, String role,
                                                         T5LinearProjectionFactory projectionFactory) {
        T5TensorData tensor = optional(weights, role);
        return tensor == null ? null : projectionFactory.create(tensor);
    }

    private static boolean[] decoderMask(int length) {
        boolean[] mask = new boolean[length];
        java.util.Arrays.fill(mask, true);
        return mask;
    }

    @Override
    public void close() {
        projectionFactory.close();
    }

    private static void validateInput(int[] decoderInputIds) {
        if (decoderInputIds == null || decoderInputIds.length == 0) {
            throw new IllegalArgumentException("decoderInputIds must not be empty");
        }
        for (int tokenId : decoderInputIds) {
            if (tokenId < 0) {
                throw new IllegalArgumentException("decoderInputIds must not contain negative token ids");
            }
        }
    }
}
