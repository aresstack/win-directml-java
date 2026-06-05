package com.aresstack.windirectml.inference.t5;

import java.util.Arrays;
import java.util.Objects;

/**
 * Reference decoder output state for the T5 decoder path.
 */
public final class T5DecoderState {
    private final int generatedTokens;
    private final int hiddenSize;
    private final float[] hiddenStates;
    private final T5DecoderCache nextCache;

    public T5DecoderState(int generatedTokens) {
        this(generatedTokens, 0, new float[0], null);
    }

    public T5DecoderState(int generatedTokens, int hiddenSize, float[] hiddenStates) {
        this(generatedTokens, hiddenSize, hiddenStates, null);
    }

    private T5DecoderState(int generatedTokens, int hiddenSize, float[] hiddenStates, T5DecoderCache nextCache) {
        if (generatedTokens < 0) {
            throw new IllegalArgumentException("generatedTokens must not be negative: " + generatedTokens);
        }
        if (hiddenSize < 0) {
            throw new IllegalArgumentException("hiddenSize must not be negative: " + hiddenSize);
        }
        this.generatedTokens = generatedTokens;
        this.hiddenSize = hiddenSize;
        this.hiddenStates = Objects.requireNonNull(hiddenStates, "hiddenStates").clone();
        this.nextCache = nextCache;
        if (generatedTokens == 0 && this.hiddenStates.length != 0) {
            throw new IllegalArgumentException("hiddenStates must be empty when generatedTokens is zero");
        }
        if (generatedTokens > 0 && hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be positive when generatedTokens is positive");
        }
        if (hiddenSize > 0 && this.hiddenStates.length != generatedTokens * hiddenSize) {
            throw new IllegalArgumentException("hiddenStates length mismatch: " + this.hiddenStates.length
                    + ", expected=" + (generatedTokens * hiddenSize));
        }
    }

    static T5DecoderState withNextCache(int generatedTokens,
                                        int hiddenSize,
                                        float[] hiddenStates,
                                        T5DecoderCache nextCache) {
        return new T5DecoderState(generatedTokens, hiddenSize, hiddenStates,
                Objects.requireNonNull(nextCache, "nextCache"));
    }

    public int generatedTokens() {
        return generatedTokens;
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    public float[] hiddenStates() {
        return hiddenStates.clone();
    }

    public float[] lastHiddenState() {
        if (generatedTokens == 0) {
            return new float[0];
        }
        int offset = (generatedTokens - 1) * hiddenSize;
        return Arrays.copyOfRange(hiddenStates, offset, offset + hiddenSize);
    }

    T5DecoderCache nextCache() {
        return nextCache;
    }
}
