package com.aresstack.windirectml.inference.t5;

import java.util.Arrays;
import java.util.Objects;

/**
 * Reference encoder output for the T5 encoder-decoder runtime path.
 */
public final class T5EncoderOutput {
    private final int inputTokens;
    private final int hiddenSize;
    private final float[] hiddenStates;
    private final boolean[] attentionMask;

    public T5EncoderOutput(int inputTokens, int hiddenSize, float[] hiddenStates, boolean[] attentionMask) {
        if (inputTokens <= 0) {
            throw new IllegalArgumentException("inputTokens must be positive: " + inputTokens);
        }
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be positive: " + hiddenSize);
        }
        this.inputTokens = inputTokens;
        this.hiddenSize = hiddenSize;
        this.hiddenStates = Objects.requireNonNull(hiddenStates, "hiddenStates").clone();
        this.attentionMask = attentionMask == null ? new boolean[inputTokens] : attentionMask.clone();
        if (this.hiddenStates.length != inputTokens * hiddenSize) {
            throw new IllegalArgumentException("hiddenStates length mismatch: " + this.hiddenStates.length
                    + ", expected=" + (inputTokens * hiddenSize));
        }
        if (this.attentionMask.length != inputTokens) {
            throw new IllegalArgumentException("attentionMask length mismatch: " + this.attentionMask.length
                    + ", expected=" + inputTokens);
        }
    }

    public int inputTokens() {
        return inputTokens;
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    public float[] hiddenStates() {
        return hiddenStates.clone();
    }

    public boolean[] attentionMask() {
        return attentionMask.clone();
    }

    public float[] tokenHiddenState(int tokenIndex) {
        if (tokenIndex < 0 || tokenIndex >= inputTokens) {
            throw new IllegalArgumentException("tokenIndex outside encoder output: " + tokenIndex);
        }
        return Arrays.copyOfRange(hiddenStates, tokenIndex * hiddenSize, tokenIndex * hiddenSize + hiddenSize);
    }
}
