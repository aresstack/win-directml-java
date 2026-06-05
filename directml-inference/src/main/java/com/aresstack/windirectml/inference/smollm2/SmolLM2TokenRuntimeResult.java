package com.aresstack.windirectml.inference.smollm2;

import java.util.List;

/**
 * Token-level result produced by the SmolLM2 reference runtime.
 */
public record SmolLM2TokenRuntimeResult(List<Integer> inputTokenIds,
                                        List<Integer> generatedTokenIds,
                                        List<Integer> fullTokenIds,
                                        int tokensGenerated,
                                        String finishReason,
                                        int maxNewTokens) {
    public SmolLM2TokenRuntimeResult {
        inputTokenIds = inputTokenIds == null ? List.of() : List.copyOf(inputTokenIds);
        generatedTokenIds = generatedTokenIds == null ? List.of() : List.copyOf(generatedTokenIds);
        fullTokenIds = fullTokenIds == null ? List.of() : List.copyOf(fullTokenIds);
        finishReason = finishReason == null ? "" : finishReason;
        if (tokensGenerated < 0) {
            throw new IllegalArgumentException("tokensGenerated must not be negative");
        }
        if (maxNewTokens < 0) {
            throw new IllegalArgumentException("maxNewTokens must not be negative");
        }
    }

    public SmolLM2TokenRuntimeResult(List<Integer> inputTokenIds,
                                     List<Integer> generatedTokenIds,
                                     List<Integer> fullTokenIds,
                                     int tokensGenerated,
                                     String finishReason) {
        this(inputTokenIds, generatedTokenIds, fullTokenIds, tokensGenerated, finishReason, tokensGenerated);
    }

    public int inputTokenCount() {
        return inputTokenIds.size();
    }

    public int fullTokenCount() {
        return fullTokenIds.size();
    }

    public boolean immediateEos() {
        return tokensGenerated == 1 && "eos_token".equals(finishReason);
    }
}
