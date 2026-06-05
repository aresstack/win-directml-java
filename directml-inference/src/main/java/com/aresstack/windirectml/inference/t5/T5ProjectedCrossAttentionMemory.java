package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * Projected K/V tensors for T5 decoder cross-attention memory.
 */
public final class T5ProjectedCrossAttentionMemory {
    private final float[] key;
    private final float[] value;

    public T5ProjectedCrossAttentionMemory(float[] key, float[] value) {
        this.key = Objects.requireNonNull(key, "key");
        this.value = Objects.requireNonNull(value, "value");
        if (key.length != value.length) {
            throw new IllegalArgumentException("Cross-attention memory projection lengths differ: k="
                    + key.length + ", v=" + value.length);
        }
    }

    public float[] key() {
        return key;
    }

    public float[] value() {
        return value;
    }
}
