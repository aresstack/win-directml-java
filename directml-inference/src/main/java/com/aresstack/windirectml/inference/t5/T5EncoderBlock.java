package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * One T5 encoder block in the reference pipeline.
 */
public final class T5EncoderBlock {
    private final T5LayerNorm selfAttentionLayerNorm;
    private final T5SelfAttention selfAttention;
    private final T5LayerNorm feedForwardLayerNorm;
    private final T5FeedForward feedForward;

    public T5EncoderBlock(T5LayerNorm selfAttentionLayerNorm,
                          T5SelfAttention selfAttention,
                          T5LayerNorm feedForwardLayerNorm,
                          T5FeedForward feedForward) {
        this.selfAttentionLayerNorm = Objects.requireNonNull(selfAttentionLayerNorm, "selfAttentionLayerNorm");
        this.selfAttention = Objects.requireNonNull(selfAttention, "selfAttention");
        this.feedForwardLayerNorm = Objects.requireNonNull(feedForwardLayerNorm, "feedForwardLayerNorm");
        this.feedForward = Objects.requireNonNull(feedForward, "feedForward");
    }

    public float[] apply(float[] hiddenStates, int sequenceLength, int hiddenSize, boolean[] attentionMask) {
        float[] normedForAttention = selfAttentionLayerNorm.applySequence(hiddenStates, sequenceLength, hiddenSize);
        float[] attentionOutput = selfAttention.apply(normedForAttention, sequenceLength, attentionMask);
        float[] afterAttention = T5ReferenceMath.add(hiddenStates, attentionOutput);
        float[] normedForFeedForward = feedForwardLayerNorm.applySequence(afterAttention, sequenceLength, hiddenSize);
        float[] feedForwardOutput = feedForward.apply(normedForFeedForward, sequenceLength);
        return T5ReferenceMath.add(afterAttention, feedForwardOutput);
    }
}
