package com.aresstack.windirectml.inference.gemma;

import java.util.Objects;

/**
 * Result of a native WARP generation (GEMMA-WARP-10b).
 *
 * @param generatedTokenIds the visible generated ids (the stop token is excluded)
 * @param fullTokenIds      prompt ids followed by {@code generatedTokenIds} (also stop-token-free)
 * @param finishReason      why generation stopped
 * @param promptTokenCount  number of prompt tokens
 * @param outputTokenCount  number of visible generated tokens ({@code = generatedTokenIds.length})
 */
public record Gemma3GenerationResult(
        int[] generatedTokenIds,
        int[] fullTokenIds,
        FinishReason finishReason,
        int promptTokenCount,
        int outputTokenCount) {

    public enum FinishReason { STOP_TOKEN, MAX_TOKENS }

    public Gemma3GenerationResult {
        Objects.requireNonNull(generatedTokenIds, "generatedTokenIds");
        Objects.requireNonNull(fullTokenIds, "fullTokenIds");
        Objects.requireNonNull(finishReason, "finishReason");
    }
}
