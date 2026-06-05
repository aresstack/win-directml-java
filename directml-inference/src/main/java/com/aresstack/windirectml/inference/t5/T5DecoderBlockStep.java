package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * Result of one incremental T5 decoder block step.
 */
public final class T5DecoderBlockStep {
    private final float[] hiddenState;
    private final T5SelfAttentionMemory selfAttentionMemory;

    public T5DecoderBlockStep(float[] hiddenState, T5SelfAttentionMemory selfAttentionMemory) {
        this.hiddenState = Objects.requireNonNull(hiddenState, "hiddenState");
        this.selfAttentionMemory = Objects.requireNonNull(selfAttentionMemory, "selfAttentionMemory");
    }

    public float[] hiddenState() {
        return hiddenState.clone();
    }

    float[] hiddenStateUnsafe() {
        return hiddenState;
    }

    public T5SelfAttentionMemory selfAttentionMemory() {
        return selfAttentionMemory;
    }
}
