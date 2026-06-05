package com.aresstack.windirectml.inference.t5;

import java.util.Arrays;
import java.util.Objects;

/**
 * Reference decoder cache boundary for T5.
 *
 * <p>v37 keeps the cache immutable and stores the generated decoder prefix.
 * It does not yet reuse key/value tensors. The WARP decoder cache can later
 * replace the internals without changing the decode-step API.</p>
 */
public final class T5DecoderCache {
    private static final T5DecoderCache EMPTY = new T5DecoderCache(new int[0], 0, new float[0]);

    private final int[] tokenIds;
    private final int hiddenSize;
    private final float[] hiddenStates;

    private T5DecoderCache(int[] tokenIds, int hiddenSize, float[] hiddenStates) {
        this.tokenIds = Objects.requireNonNull(tokenIds, "tokenIds").clone();
        this.hiddenSize = hiddenSize;
        this.hiddenStates = Objects.requireNonNull(hiddenStates, "hiddenStates").clone();
        if (hiddenSize < 0) {
            throw new IllegalArgumentException("hiddenSize must not be negative: " + hiddenSize);
        }
        if (hiddenSize == 0 && this.hiddenStates.length != 0) {
            throw new IllegalArgumentException("hiddenStates require a positive hiddenSize");
        }
        if (hiddenSize > 0 && this.hiddenStates.length != this.tokenIds.length * hiddenSize) {
            throw new IllegalArgumentException("hiddenStates length mismatch: " + this.hiddenStates.length
                    + ", expected=" + (this.tokenIds.length * hiddenSize));
        }
    }

    public static T5DecoderCache empty() {
        return EMPTY;
    }

    public static T5DecoderCache fromTokenIds(int[] tokenIds) {
        Objects.requireNonNull(tokenIds, "tokenIds");
        return new T5DecoderCache(tokenIds, 0, new float[0]);
    }

    public T5DecoderCache append(int tokenId, T5DecoderState state) {
        Objects.requireNonNull(state, "state");
        if (state.generatedTokens() != generatedTokens() + 1) {
            throw new IllegalArgumentException("Decoder state must contain exactly one appended token: state="
                    + state.generatedTokens() + ", cache=" + generatedTokens());
        }
        int[] nextTokens = Arrays.copyOf(tokenIds, tokenIds.length + 1);
        nextTokens[nextTokens.length - 1] = tokenId;
        return new T5DecoderCache(nextTokens, state.hiddenSize(), state.hiddenStates());
    }

    public int generatedTokens() {
        return tokenIds.length;
    }

    public boolean isEmpty() {
        return tokenIds.length == 0;
    }

    public int[] tokenIds() {
        return tokenIds.clone();
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    public float[] hiddenStates() {
        return hiddenStates.clone();
    }

    int[] withAppendedToken(int tokenId) {
        int[] next = Arrays.copyOf(tokenIds, tokenIds.length + 1);
        next[next.length - 1] = tokenId;
        return next;
    }
}
