package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * Result of one incremental decoder self-attention step.
 */
public final class T5SelfAttentionStep {
    private final float[] output;
    private final T5SelfAttentionMemory memory;

    public T5SelfAttentionStep(float[] output, T5SelfAttentionMemory memory) {
        this.output = Objects.requireNonNull(output, "output");
        this.memory = Objects.requireNonNull(memory, "memory");
    }

    public float[] output() {
        return output.clone();
    }

    float[] outputUnsafe() {
        return output;
    }

    public T5SelfAttentionMemory memory() {
        return memory;
    }
}
