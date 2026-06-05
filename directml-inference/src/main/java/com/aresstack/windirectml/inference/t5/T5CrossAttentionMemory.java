package com.aresstack.windirectml.inference.t5;

import java.util.Arrays;
import java.util.Objects;

/**
 * Precomputed encoder-side key/value tensors for one T5 decoder cross-attention layer.
 *
 * <p>The encoder output is constant for the whole generation. Keep its cross-attention
 * K/V projections out of the per-token decoder loop so WARP projections can run once
 * per decoder layer instead of once per generated token.</p>
 */
public final class T5CrossAttentionMemory {
    private final int encoderLength;
    private final int innerSize;
    private final boolean[] attentionMask;
    private final float[] key;
    private final float[] value;

    public T5CrossAttentionMemory(int encoderLength,
                                  int innerSize,
                                  boolean[] attentionMask,
                                  float[] key,
                                  float[] value) {
        if (encoderLength <= 0) {
            throw new IllegalArgumentException("encoderLength must be positive: " + encoderLength);
        }
        if (innerSize <= 0) {
            throw new IllegalArgumentException("innerSize must be positive: " + innerSize);
        }
        this.encoderLength = encoderLength;
        this.innerSize = innerSize;
        this.attentionMask = Objects.requireNonNull(attentionMask, "attentionMask").clone();
        this.key = Objects.requireNonNull(key, "key").clone();
        this.value = Objects.requireNonNull(value, "value").clone();
        validateLengths();
    }

    public int encoderLength() {
        return encoderLength;
    }

    public int innerSize() {
        return innerSize;
    }

    public boolean attentionEnabled(int tokenIndex) {
        if (tokenIndex < 0 || tokenIndex >= attentionMask.length) {
            throw new IllegalArgumentException("tokenIndex outside encoder mask: " + tokenIndex);
        }
        return attentionMask[tokenIndex];
    }

    public float keyAt(int tokenIndex, int headOffset, int dim) {
        return key[tokenIndex * innerSize + headOffset + dim];
    }

    public float valueAt(int tokenIndex, int headOffset, int dim) {
        return value[tokenIndex * innerSize + headOffset + dim];
    }

    float[] keyUnsafe() {
        return key;
    }

    float[] valueUnsafe() {
        return value;
    }

    private void validateLengths() {
        if (attentionMask.length != encoderLength) {
            throw new IllegalArgumentException("attentionMask length mismatch: " + attentionMask.length
                    + ", expected=" + encoderLength);
        }
        int expected = encoderLength * innerSize;
        if (key.length != expected) {
            throw new IllegalArgumentException("key length mismatch: " + key.length + ", expected=" + expected);
        }
        if (value.length != expected) {
            throw new IllegalArgumentException("value length mismatch: " + value.length + ", expected=" + expected);
        }
    }

    @Override
    public String toString() {
        return "T5CrossAttentionMemory{" +
                "encoderLength=" + encoderLength +
                ", innerSize=" + innerSize +
                ", mask=" + Arrays.toString(attentionMask) +
                '}';
    }
}
