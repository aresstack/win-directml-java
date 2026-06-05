package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * Cross-attention memory projection backed by independent K and V projections.
 */
public final class T5SplitCrossAttentionMemoryProjection implements T5CrossAttentionMemoryProjection {
    private final T5LinearProjection keyProjection;
    private final T5LinearProjection valueProjection;

    public T5SplitCrossAttentionMemoryProjection(T5LinearProjection keyProjection, T5LinearProjection valueProjection) {
        this.keyProjection = Objects.requireNonNull(keyProjection, "keyProjection");
        this.valueProjection = Objects.requireNonNull(valueProjection, "valueProjection");
    }

    @Override
    public T5ProjectedCrossAttentionMemory applySequence(float[] encoderHiddenStates, int encoderLength, int hiddenSize) {
        return new T5ProjectedCrossAttentionMemory(
                keyProjection.applySequence(encoderHiddenStates, encoderLength, hiddenSize),
                valueProjection.applySequence(encoderHiddenStates, encoderLength, hiddenSize));
    }
}
