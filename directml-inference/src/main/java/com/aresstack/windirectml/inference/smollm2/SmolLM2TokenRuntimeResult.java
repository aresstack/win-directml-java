package com.aresstack.windirectml.inference.smollm2;

import java.util.List;

/**
 * Token-level result produced by the SmolLM2 reference runtime.
 */
public record SmolLM2TokenRuntimeResult(List<Integer> inputTokenIds,
                                        List<Integer> generatedTokenIds,
                                        List<Integer> fullTokenIds,
                                        int tokensGenerated,
                                        String finishReason) {
    public SmolLM2TokenRuntimeResult {
        inputTokenIds = inputTokenIds == null ? List.of() : List.copyOf(inputTokenIds);
        generatedTokenIds = generatedTokenIds == null ? List.of() : List.copyOf(generatedTokenIds);
        fullTokenIds = fullTokenIds == null ? List.of() : List.copyOf(fullTokenIds);
        finishReason = finishReason == null ? "" : finishReason;
    }
}
