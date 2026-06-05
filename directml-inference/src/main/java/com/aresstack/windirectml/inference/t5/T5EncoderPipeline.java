package com.aresstack.windirectml.inference.t5;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * CPU reference T5 encoder pipeline.
 *
 * <p>Use this implementation to validate package loading, tensor roles, and
 * seq2seq encoder semantics before replacing the math with WARP kernels.</p>
 */
public final class T5EncoderPipeline implements T5EncoderRunner, AutoCloseable {
    private final T5PackageMetadata metadata;
    private final T5Weights weights;
    private final T5TensorData sharedEmbedding;
    private final List<T5EncoderBlock> blocks;
    private final T5LayerNorm finalLayerNorm;
    private final T5LinearProjectionFactory projectionFactory;

    private T5EncoderPipeline(T5PackageMetadata metadata,
                              T5Weights weights,
                              T5TensorData sharedEmbedding,
                              List<T5EncoderBlock> blocks,
                              T5LayerNorm finalLayerNorm,
                              T5LinearProjectionFactory projectionFactory) {
        this.metadata = Objects.requireNonNull(metadata, "metadata");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.sharedEmbedding = Objects.requireNonNull(sharedEmbedding, "sharedEmbedding");
        this.blocks = List.copyOf(blocks);
        this.finalLayerNorm = Objects.requireNonNull(finalLayerNorm, "finalLayerNorm");
        this.projectionFactory = Objects.requireNonNull(projectionFactory, "projectionFactory");
    }

    public static T5EncoderPipeline from(T5Weights weights) {
        Objects.requireNonNull(weights, "weights");
        return from(weights, T5ReferenceLinearProjectionFactory.INSTANCE);
    }

    public static T5EncoderPipeline from(T5Weights weights, T5LinearProjectionFactory projectionFactory) {
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(projectionFactory, "projectionFactory");
        T5PackageMetadata metadata = weights.metadata();
        T5TensorData sharedEmbedding = T5TensorData.from(weights.sharedEmbedding());
        List<T5EncoderBlock> blocks = new ArrayList<>();
        T5TensorData encoderBias = optional(weights, weights.encoderRelativeAttentionBias() == null
                ? null : "encoder.relative_attention_bias");
        T5RelativePositionBias relativePositionBias = new T5RelativePositionBias(
                encoderBias, metadata.relativeAttentionBuckets(), metadata.relativeAttentionMaxDistance(), metadata.numHeads());
        for (int layer = 0; layer < metadata.encoderLayers(); layer++) {
            blocks.add(createBlock(weights, metadata, layer, relativePositionBias, projectionFactory));
        }
        T5LayerNorm finalLayerNorm = new T5LayerNorm(T5TensorData.from(weights.encoderFinalLayerNorm()), 1e-6f);
        return new T5EncoderPipeline(metadata, weights, sharedEmbedding, blocks, finalLayerNorm, projectionFactory);
    }

    @Override
    public T5EncoderOutput encode(int[] inputTokenIds) {
        return encode(inputTokenIds, null);
    }

    @Override
    public T5EncoderOutput encode(int[] inputTokenIds, boolean[] attentionMask) {
        validateInput(inputTokenIds);
        boolean[] mask = normalizeMask(inputTokenIds, attentionMask);
        int sequenceLength = inputTokenIds.length;
        int hiddenSize = metadata.dModel();
        float[] hiddenStates = embed(inputTokenIds, mask);
        for (T5EncoderBlock block : blocks) {
            hiddenStates = block.apply(hiddenStates, sequenceLength, hiddenSize, mask);
        }
        hiddenStates = finalLayerNorm.applySequence(hiddenStates, sequenceLength, hiddenSize);
        return new T5EncoderOutput(sequenceLength, hiddenSize, hiddenStates, mask);
    }

    public T5Weights weights() {
        return weights;
    }

    @Override
    public String executionMode() {
        return "reference-encoder";
    }


    private float[] embed(int[] inputTokenIds, boolean[] mask) {
        int sequenceLength = inputTokenIds.length;
        int hiddenSize = metadata.dModel();
        float[] result = new float[sequenceLength * hiddenSize];
        for (int token = 0; token < sequenceLength; token++) {
            int tokenId = inputTokenIds[token];
            if (tokenId >= sharedEmbedding.dim(0)) {
                throw new IllegalArgumentException("inputTokenIds contains token outside vocabulary: " + tokenId);
            }
            if (!mask[token]) {
                continue;
            }
            for (int dim = 0; dim < hiddenSize; dim++) {
                result[token * hiddenSize + dim] = sharedEmbedding.at(tokenId, dim);
            }
        }
        return result;
    }

    private static T5EncoderBlock createBlock(T5Weights weights,
                                              T5PackageMetadata metadata,
                                              int layer,
                                              T5RelativePositionBias relativePositionBias,
                                              T5LinearProjectionFactory projectionFactory) {
        T5LayerNorm attentionLayerNorm = new T5LayerNorm(
                T5TensorData.from(weights.encoderLayerNorm(layer, 0)), 1e-6f);
        T5SelfAttention selfAttention = new T5SelfAttention(metadata,
                projectionFactory.createSelfAttentionProjection(
                        T5TensorData.from(weights.encoderSelfAttention(layer, "q")),
                        T5TensorData.from(weights.encoderSelfAttention(layer, "k")),
                        T5TensorData.from(weights.encoderSelfAttention(layer, "v"))),
                projectionFactory.create(T5TensorData.from(weights.encoderSelfAttention(layer, "o"))),
                relativePositionBias, true, false);
        T5LayerNorm feedForwardLayerNorm = new T5LayerNorm(
                T5TensorData.from(weights.encoderLayerNorm(layer, 1)), 1e-6f);
        T5FeedForward feedForward = new T5FeedForward(metadata,
                optionalProjection(weights, "encoder.layer." + layer + ".ffn.wi", projectionFactory),
                optionalProjection(weights, "encoder.layer." + layer + ".ffn.wi_0", projectionFactory),
                optionalProjection(weights, "encoder.layer." + layer + ".ffn.wi_1", projectionFactory),
                projectionFactory.create(T5TensorData.from(weights.encoderFeedForward(layer, "wo"))));
        return new T5EncoderBlock(attentionLayerNorm, selfAttention, feedForwardLayerNorm, feedForward);
    }

    private static T5TensorData optional(T5Weights weights, String role) {
        if (role == null) {
            return null;
        }
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

    private boolean[] normalizeMask(int[] inputTokenIds, boolean[] attentionMask) {
        if (attentionMask != null) {
            if (attentionMask.length != inputTokenIds.length) {
                throw new IllegalArgumentException("attentionMask length mismatch: "
                        + attentionMask.length + " != " + inputTokenIds.length);
            }
            return java.util.Arrays.copyOf(attentionMask, attentionMask.length);
        }
        boolean[] mask = new boolean[inputTokenIds.length];
        int pad = metadata.specialTokens().padTokenId();
        for (int i = 0; i < inputTokenIds.length; i++) {
            mask[i] = inputTokenIds[i] != pad;
        }
        return mask;
    }

    @Override
    public void close() {
        projectionFactory.close();
    }

    private static void validateInput(int[] inputTokenIds) {
        if (inputTokenIds == null || inputTokenIds.length == 0) {
            throw new IllegalArgumentException("inputTokenIds must not be empty");
        }
        for (int tokenId : inputTokenIds) {
            if (tokenId < 0) {
                throw new IllegalArgumentException("inputTokenIds must not contain negative token ids");
            }
        }
    }
}
