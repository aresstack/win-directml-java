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
public final class T5DecoderPipeline {
    private final T5PackageMetadata metadata;
    private final T5Weights weights;
    private final T5TensorData sharedEmbedding;
    private final List<T5DecoderBlock> blocks;
    private final T5LayerNorm finalLayerNorm;

    private T5DecoderPipeline(T5PackageMetadata metadata,
                              T5Weights weights,
                              T5TensorData sharedEmbedding,
                              List<T5DecoderBlock> blocks,
                              T5LayerNorm finalLayerNorm) {
        this.metadata = Objects.requireNonNull(metadata, "metadata");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.sharedEmbedding = Objects.requireNonNull(sharedEmbedding, "sharedEmbedding");
        this.blocks = List.copyOf(blocks);
        this.finalLayerNorm = Objects.requireNonNull(finalLayerNorm, "finalLayerNorm");
    }

    public static T5DecoderPipeline from(T5Weights weights) {
        Objects.requireNonNull(weights, "weights");
        T5PackageMetadata metadata = weights.metadata();
        T5TensorData sharedEmbedding = T5TensorData.from(weights.sharedEmbedding());
        T5TensorData decoderBias = optional(weights, "decoder.relative_attention_bias");
        T5RelativePositionBias relativePositionBias = new T5RelativePositionBias(
                decoderBias, metadata.relativeAttentionBuckets(), metadata.relativeAttentionMaxDistance(), metadata.numHeads());
        List<T5DecoderBlock> blocks = new ArrayList<>();
        for (int layer = 0; layer < metadata.decoderLayers(); layer++) {
            blocks.add(createBlock(weights, metadata, layer, relativePositionBias));
        }
        T5LayerNorm finalLayerNorm = new T5LayerNorm(T5TensorData.from(weights.decoderFinalLayerNorm()), 1e-6f);
        return new T5DecoderPipeline(metadata, weights, sharedEmbedding, blocks, finalLayerNorm);
    }

    public T5DecoderState decode(int[] decoderInputIds, T5EncoderOutput encoderOutput) {
        validateInput(decoderInputIds);
        Objects.requireNonNull(encoderOutput, "encoderOutput");
        boolean[] decoderMask = decoderMask(decoderInputIds.length);
        int sequenceLength = decoderInputIds.length;
        int hiddenSize = metadata.dModel();
        float[] hiddenStates = embed(decoderInputIds);
        for (T5DecoderBlock block : blocks) {
            hiddenStates = block.apply(hiddenStates, sequenceLength, hiddenSize, decoderMask, encoderOutput);
        }
        hiddenStates = finalLayerNorm.applySequence(hiddenStates, sequenceLength, hiddenSize);
        return new T5DecoderState(sequenceLength, hiddenSize, hiddenStates);
    }

    public T5DecoderState decodeStep(int decoderTokenId, T5EncoderOutput encoderOutput, T5DecoderCache cache) {
        T5DecoderCache safeCache = cache == null ? T5DecoderCache.empty() : cache;
        return decode(safeCache.withAppendedToken(decoderTokenId), encoderOutput);
    }

    public T5Weights weights() {
        return weights;
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
                                              T5RelativePositionBias relativePositionBias) {
        T5LayerNorm selfAttentionLayerNorm = new T5LayerNorm(
                T5TensorData.from(weights.decoderLayerNorm(layer, 0)), 1e-6f);
        T5SelfAttention selfAttention = new T5SelfAttention(metadata,
                T5TensorData.from(weights.decoderSelfAttention(layer, "q")),
                T5TensorData.from(weights.decoderSelfAttention(layer, "k")),
                T5TensorData.from(weights.decoderSelfAttention(layer, "v")),
                T5TensorData.from(weights.decoderSelfAttention(layer, "o")),
                relativePositionBias,
                false,
                true);
        T5LayerNorm crossAttentionLayerNorm = new T5LayerNorm(
                T5TensorData.from(weights.decoderLayerNorm(layer, 1)), 1e-6f);
        T5CrossAttention crossAttention = new T5CrossAttention(metadata,
                T5TensorData.from(weights.decoderCrossAttention(layer, "q")),
                T5TensorData.from(weights.decoderCrossAttention(layer, "k")),
                T5TensorData.from(weights.decoderCrossAttention(layer, "v")),
                T5TensorData.from(weights.decoderCrossAttention(layer, "o")));
        T5LayerNorm feedForwardLayerNorm = new T5LayerNorm(
                T5TensorData.from(weights.decoderLayerNorm(layer, 2)), 1e-6f);
        T5FeedForward feedForward = new T5FeedForward(metadata,
                optional(weights, "decoder.layer." + layer + ".ffn.wi"),
                optional(weights, "decoder.layer." + layer + ".ffn.wi_0"),
                optional(weights, "decoder.layer." + layer + ".ffn.wi_1"),
                T5TensorData.from(weights.decoderFeedForward(layer, "wo")));
        return new T5DecoderBlock(selfAttentionLayerNorm, selfAttention, crossAttentionLayerNorm,
                crossAttention, feedForwardLayerNorm, feedForward);
    }

    private static T5TensorData optional(T5Weights weights, String role) {
        if (weights.optional(role) == null) {
            return null;
        }
        return T5TensorData.from(weights.optional(role));
    }

    private static boolean[] decoderMask(int length) {
        boolean[] mask = new boolean[length];
        java.util.Arrays.fill(mask, true);
        return mask;
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
