package com.aresstack.windirectml.inference.smollm2;

import java.util.List;

/**
 * Result contract for future SmolLM2 generation.
 */
public record SmolLM2RuntimeResult(String generatedText,
                                   List<Integer> generatedTokenIds,
                                   int tokensGenerated,
                                   String finishReason) {
    public SmolLM2RuntimeResult {
        generatedText = generatedText == null ? "" : generatedText;
        generatedTokenIds = generatedTokenIds == null ? List.of() : List.copyOf(generatedTokenIds);
        finishReason = finishReason == null ? "" : finishReason;
    }
}
