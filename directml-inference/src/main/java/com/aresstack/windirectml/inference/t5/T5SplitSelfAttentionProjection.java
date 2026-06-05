package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * Self-attention Q/K/V projection backed by three independent linear projections.
 */
public final class T5SplitSelfAttentionProjection implements T5SelfAttentionProjection {
    private final T5LinearProjection queryProjection;
    private final T5LinearProjection keyProjection;
    private final T5LinearProjection valueProjection;

    public T5SplitSelfAttentionProjection(T5LinearProjection queryProjection,
                                          T5LinearProjection keyProjection,
                                          T5LinearProjection valueProjection) {
        this.queryProjection = Objects.requireNonNull(queryProjection, "queryProjection");
        this.keyProjection = Objects.requireNonNull(keyProjection, "keyProjection");
        this.valueProjection = Objects.requireNonNull(valueProjection, "valueProjection");
    }

    @Override
    public T5ProjectedSelfAttention applySequence(float[] hiddenStates, int sequenceLength, int hiddenSize) {
        return new T5ProjectedSelfAttention(
                queryProjection.applySequence(hiddenStates, sequenceLength, hiddenSize),
                keyProjection.applySequence(hiddenStates, sequenceLength, hiddenSize),
                valueProjection.applySequence(hiddenStates, sequenceLength, hiddenSize));
    }

    @Override
    public T5ProjectedSelfAttention apply(float[] hiddenState) {
        return new T5ProjectedSelfAttention(
                queryProjection.apply(hiddenState),
                keyProjection.apply(hiddenState),
                valueProjection.apply(hiddenState));
    }
}
