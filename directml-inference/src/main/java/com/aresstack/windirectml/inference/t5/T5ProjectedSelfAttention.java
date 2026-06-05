package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * Projected Q/K/V tensors for one T5 self-attention invocation.
 */
public final class T5ProjectedSelfAttention {
    private final float[] query;
    private final float[] key;
    private final float[] value;

    public T5ProjectedSelfAttention(float[] query, float[] key, float[] value) {
        this.query = Objects.requireNonNull(query, "query");
        this.key = Objects.requireNonNull(key, "key");
        this.value = Objects.requireNonNull(value, "value");
        if (query.length != key.length || query.length != value.length) {
            throw new IllegalArgumentException("Self-attention projection lengths differ: q="
                    + query.length + ", k=" + key.length + ", v=" + value.length);
        }
    }

    public float[] query() {
        return query;
    }

    public float[] key() {
        return key;
    }

    public float[] value() {
        return value;
    }
}
