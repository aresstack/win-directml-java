package com.aresstack.windirectml.inference.t5;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Immutable decoder cache boundary for T5.
 *
 * <p>The cache stores the generated decoder prefix, the final decoder hidden
 * states used by the LM head, and the per-layer decoder self-attention K/V
 * memory needed by incremental decoding.</p>
 */
public final class T5DecoderCache {
    private static final T5DecoderCache EMPTY = new T5DecoderCache(new int[0], 0, new float[0], Collections.emptyList());

    private final int[] tokenIds;
    private final int hiddenSize;
    private final float[] hiddenStates;
    private final List<T5SelfAttentionMemory> selfAttentionMemories;

    private T5DecoderCache(int[] tokenIds,
                           int hiddenSize,
                           float[] hiddenStates,
                           List<T5SelfAttentionMemory> selfAttentionMemories) {
        this.tokenIds = Objects.requireNonNull(tokenIds, "tokenIds").clone();
        this.hiddenSize = hiddenSize;
        this.hiddenStates = Objects.requireNonNull(hiddenStates, "hiddenStates").clone();
        this.selfAttentionMemories = Collections.unmodifiableList(new ArrayList<>(
                Objects.requireNonNull(selfAttentionMemories, "selfAttentionMemories")));
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
        for (T5SelfAttentionMemory memory : this.selfAttentionMemories) {
            if (memory.sequenceLength() != this.tokenIds.length) {
                throw new IllegalArgumentException("Self-attention memory length mismatch: "
                        + memory.sequenceLength() + " != " + this.tokenIds.length);
            }
        }
    }

    public static T5DecoderCache empty() {
        return EMPTY;
    }

    public static T5DecoderCache fromTokenIds(int[] tokenIds) {
        Objects.requireNonNull(tokenIds, "tokenIds");
        return new T5DecoderCache(tokenIds, 0, new float[0], Collections.emptyList());
    }

    static T5DecoderCache fromIncrementalState(int[] tokenIds,
                                               int hiddenSize,
                                               float[] hiddenStates,
                                               List<T5SelfAttentionMemory> selfAttentionMemories) {
        return new T5DecoderCache(tokenIds, hiddenSize, hiddenStates, selfAttentionMemories);
    }

    public T5DecoderCache append(int tokenId, T5DecoderState state) {
        Objects.requireNonNull(state, "state");
        T5DecoderCache incremental = state.nextCache();
        if (incremental != null) {
            validateIncrementalAppend(tokenId, incremental);
            return incremental;
        }
        if (state.generatedTokens() != generatedTokens() + 1) {
            throw new IllegalArgumentException("Decoder state must contain exactly one appended token: state="
                    + state.generatedTokens() + ", cache=" + generatedTokens());
        }
        int[] nextTokens = Arrays.copyOf(tokenIds, tokenIds.length + 1);
        nextTokens[nextTokens.length - 1] = tokenId;
        return new T5DecoderCache(nextTokens, state.hiddenSize(), state.hiddenStates(), Collections.emptyList());
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

    boolean hasSelfAttentionMemories(int layerCount) {
        return selfAttentionMemories.size() == layerCount;
    }

    T5SelfAttentionMemory selfAttentionMemory(int layer) {
        if (selfAttentionMemories.isEmpty()) {
            return T5SelfAttentionMemory.empty();
        }
        return selfAttentionMemories.get(layer);
    }

    int[] withAppendedToken(int tokenId) {
        int[] next = Arrays.copyOf(tokenIds, tokenIds.length + 1);
        next[next.length - 1] = tokenId;
        return next;
    }

    private void validateIncrementalAppend(int tokenId, T5DecoderCache incremental) {
        if (incremental.generatedTokens() != generatedTokens() + 1) {
            throw new IllegalArgumentException("Incremental cache must contain exactly one appended token: next="
                    + incremental.generatedTokens() + ", cache=" + generatedTokens());
        }
        int[] nextTokens = incremental.tokenIds();
        if (nextTokens[nextTokens.length - 1] != tokenId) {
            throw new IllegalArgumentException("Incremental cache appended token mismatch: next="
                    + nextTokens[nextTokens.length - 1] + ", expected=" + tokenId);
        }
    }
}
