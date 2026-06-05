package com.aresstack.windirectml.inference.t5;

import java.util.Arrays;
import java.util.Objects;

/**
 * Cached decoder self-attention key/value memory for one T5 decoder layer.
 */
public final class T5SelfAttentionMemory {
    private static final T5SelfAttentionMemory EMPTY = new T5SelfAttentionMemory(0, 0, new float[0], new float[0], false);

    private final int sequenceLength;
    private final int innerSize;
    private final float[] key;
    private final float[] value;

    private T5SelfAttentionMemory(int sequenceLength, int innerSize, float[] key, float[] value) {
        this(sequenceLength, innerSize, key, value, true);
    }

    private T5SelfAttentionMemory(int sequenceLength, int innerSize, float[] key, float[] value, boolean copy) {
        if (sequenceLength < 0) {
            throw new IllegalArgumentException("sequenceLength must not be negative: " + sequenceLength);
        }
        if (innerSize < 0) {
            throw new IllegalArgumentException("innerSize must not be negative: " + innerSize);
        }
        this.sequenceLength = sequenceLength;
        this.innerSize = innerSize;
        this.key = copy ? Objects.requireNonNull(key, "key").clone() : Objects.requireNonNull(key, "key");
        this.value = copy ? Objects.requireNonNull(value, "value").clone() : Objects.requireNonNull(value, "value");
        int expected = sequenceLength * innerSize;
        if (this.key.length != expected) {
            throw new IllegalArgumentException("key length mismatch: " + this.key.length + ", expected=" + expected);
        }
        if (this.value.length != expected) {
            throw new IllegalArgumentException("value length mismatch: " + this.value.length + ", expected=" + expected);
        }
    }

    public static T5SelfAttentionMemory empty() {
        return EMPTY;
    }

    public static T5SelfAttentionMemory of(int sequenceLength, int innerSize, float[] key, float[] value) {
        return new T5SelfAttentionMemory(sequenceLength, innerSize, key, value);
    }

    public T5SelfAttentionMemory append(float[] tokenKey, float[] tokenValue) {
        Objects.requireNonNull(tokenKey, "tokenKey");
        Objects.requireNonNull(tokenValue, "tokenValue");
        int nextInnerSize = innerSize == 0 ? tokenKey.length : innerSize;
        if (tokenKey.length != nextInnerSize) {
            throw new IllegalArgumentException("tokenKey length mismatch: " + tokenKey.length + " != " + nextInnerSize);
        }
        if (tokenValue.length != nextInnerSize) {
            throw new IllegalArgumentException("tokenValue length mismatch: " + tokenValue.length + " != " + nextInnerSize);
        }
        float[] nextKey = Arrays.copyOf(key, key.length + nextInnerSize);
        float[] nextValue = Arrays.copyOf(value, value.length + nextInnerSize);
        System.arraycopy(tokenKey, 0, nextKey, key.length, nextInnerSize);
        System.arraycopy(tokenValue, 0, nextValue, value.length, nextInnerSize);
        return new T5SelfAttentionMemory(sequenceLength + 1, nextInnerSize, nextKey, nextValue, false);
    }

    public int sequenceLength() {
        return sequenceLength;
    }

    public int innerSize() {
        return innerSize;
    }

    public float keyAt(int token, int headOffset, int dim) {
        return key[token * innerSize + headOffset + dim];
    }

    public float valueAt(int token, int headOffset, int dim) {
        return value[token * innerSize + headOffset + dim];
    }

    public float[] key() {
        return key.clone();
    }

    public float[] value() {
        return value.clone();
    }
}
