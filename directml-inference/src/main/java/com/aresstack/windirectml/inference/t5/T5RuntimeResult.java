package com.aresstack.windirectml.inference.t5;

import java.util.Arrays;
import java.util.Objects;

/**
 * Result boundary for the future T5 WARP generation runtime.
 */
public final class T5RuntimeResult {
    public enum FinishReason {
        unsupported,
        stop_token,
        max_tokens
    }

    private final int[] outputTokenIds;
    private final FinishReason finishReason;
    private final int generatedTokens;

    public T5RuntimeResult(int[] outputTokenIds, FinishReason finishReason, int generatedTokens) {
        this.outputTokenIds = outputTokenIds == null ? new int[0] : Arrays.copyOf(outputTokenIds, outputTokenIds.length);
        this.finishReason = Objects.requireNonNull(finishReason, "finishReason");
        if (generatedTokens < 0) {
            throw new IllegalArgumentException("generatedTokens must not be negative: " + generatedTokens);
        }
        this.generatedTokens = generatedTokens;
    }

    public int[] outputTokenIds() {
        return Arrays.copyOf(outputTokenIds, outputTokenIds.length);
    }

    public FinishReason finishReason() {
        return finishReason;
    }

    public int generatedTokens() {
        return generatedTokens;
    }
}
