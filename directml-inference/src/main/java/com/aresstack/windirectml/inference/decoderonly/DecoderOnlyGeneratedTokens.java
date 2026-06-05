package com.aresstack.windirectml.inference.decoderonly;

import java.util.Arrays;

/**
 * Stores generated token IDs without reallocating on the hot decode path.
 */
public final class DecoderOnlyGeneratedTokens {

    private final int[] tokenIds;
    private int count;

    public DecoderOnlyGeneratedTokens(int capacity) {
        if (capacity < 0) {
            throw new IllegalArgumentException("capacity must not be negative");
        }
        this.tokenIds = new int[capacity];
        this.count = 0;
    }

    public void add(int tokenId) {
        if (count >= tokenIds.length) {
            throw new IllegalStateException("generated token buffer is full");
        }
        tokenIds[count] = tokenId;
        count++;
    }

    public int count() {
        return count;
    }

    public boolean isEmpty() {
        return count == 0;
    }

    public int capacity() {
        return tokenIds.length;
    }

    /**
     * Return the mutable backing array for hot-path helpers that accept a count.
     */
    public int[] backingArray() {
        return tokenIds;
    }

    public int[] copyTokenIds() {
        return Arrays.copyOf(tokenIds, count);
    }
}
