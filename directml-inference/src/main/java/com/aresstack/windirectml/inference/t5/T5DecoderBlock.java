package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * One T5 decoder block in the reference pipeline.
 */
public final class T5DecoderBlock {
    private final T5LayerNorm selfAttentionLayerNorm;
    private final T5SelfAttention selfAttention;
    private final T5LayerNorm crossAttentionLayerNorm;
    private final T5CrossAttention crossAttention;
    private final T5LayerNorm feedForwardLayerNorm;
    private final T5FeedForward feedForward;

    public T5DecoderBlock(T5LayerNorm selfAttentionLayerNorm,
                          T5SelfAttention selfAttention,
                          T5LayerNorm crossAttentionLayerNorm,
                          T5CrossAttention crossAttention,
                          T5LayerNorm feedForwardLayerNorm,
                          T5FeedForward feedForward) {
        this.selfAttentionLayerNorm = Objects.requireNonNull(selfAttentionLayerNorm, "selfAttentionLayerNorm");
        this.selfAttention = Objects.requireNonNull(selfAttention, "selfAttention");
        this.crossAttentionLayerNorm = Objects.requireNonNull(crossAttentionLayerNorm, "crossAttentionLayerNorm");
        this.crossAttention = Objects.requireNonNull(crossAttention, "crossAttention");
        this.feedForwardLayerNorm = Objects.requireNonNull(feedForwardLayerNorm, "feedForwardLayerNorm");
        this.feedForward = Objects.requireNonNull(feedForward, "feedForward");
    }

    public T5CrossAttentionMemory prepareCrossAttentionMemory(T5EncoderOutput encoderOutput) {
        return crossAttention.prepareMemory(encoderOutput);
    }

    public float[] apply(float[] hiddenStates,
                         int sequenceLength,
                         int hiddenSize,
                         boolean[] decoderAttentionMask,
                         T5CrossAttentionMemory crossAttentionMemory) {
        float[] normedForSelfAttention = selfAttentionLayerNorm.applySequence(hiddenStates, sequenceLength, hiddenSize);
        float[] selfAttentionOutput = selfAttention.apply(normedForSelfAttention, sequenceLength, decoderAttentionMask);
        float[] afterSelfAttention = T5ReferenceMath.add(hiddenStates, selfAttentionOutput);
        float[] normedForCrossAttention = crossAttentionLayerNorm.applySequence(afterSelfAttention, sequenceLength, hiddenSize);
        float[] crossAttentionOutput = crossAttention.apply(normedForCrossAttention, sequenceLength, crossAttentionMemory);
        float[] afterCrossAttention = T5ReferenceMath.add(afterSelfAttention, crossAttentionOutput);
        float[] normedForFeedForward = feedForwardLayerNorm.applySequence(afterCrossAttention, sequenceLength, hiddenSize);
        float[] feedForwardOutput = feedForward.apply(normedForFeedForward, sequenceLength);
        return T5ReferenceMath.add(afterCrossAttention, feedForwardOutput);
    }

    public T5DecoderBlockStep applyStep(float[] hiddenState,
                                        int tokenIndex,
                                        T5SelfAttentionMemory selfAttentionMemory,
                                        T5CrossAttentionMemory crossAttentionMemory) {
        int hiddenSize = hiddenState.length;
        float[] normedForSelfAttention = selfAttentionLayerNorm.applySequence(hiddenState, 1, hiddenSize);
        T5SelfAttentionStep selfAttentionStep = selfAttention.applyStep(normedForSelfAttention,
                tokenIndex, selfAttentionMemory);
        float[] afterSelfAttention = T5ReferenceMath.add(hiddenState, selfAttentionStep.output());
        float[] normedForCrossAttention = crossAttentionLayerNorm.applySequence(afterSelfAttention, 1, hiddenSize);
        float[] crossAttentionOutput = crossAttention.apply(normedForCrossAttention, 1, crossAttentionMemory);
        float[] afterCrossAttention = T5ReferenceMath.add(afterSelfAttention, crossAttentionOutput);
        float[] normedForFeedForward = feedForwardLayerNorm.applySequence(afterCrossAttention, 1, hiddenSize);
        float[] feedForwardOutput = feedForward.apply(normedForFeedForward, 1);
        return new T5DecoderBlockStep(T5ReferenceMath.add(afterCrossAttention, feedForwardOutput),
                selfAttentionStep.memory());
    }

}
