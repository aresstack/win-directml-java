package com.aresstack.windirectml.inference.smollm2;

import java.util.List;

/**
 * Result contract for future SmolLM2 generation.
 */
public record SmolLM2RuntimeResult(String generatedText,
                                   List<Integer> generatedTokenIds,
                                   int tokensGenerated,
                                   String finishReason,
                                   SmolLM2GenerationDiagnostics diagnostics) {
    public SmolLM2RuntimeResult {
        generatedText = generatedText == null ? "" : generatedText;
        generatedTokenIds = generatedTokenIds == null ? List.of() : List.copyOf(generatedTokenIds);
        finishReason = finishReason == null ? "" : finishReason;
        diagnostics = diagnostics == null
                ? new SmolLM2GenerationDiagnostics(0, tokensGenerated, generatedTokenIds.size(),
                tokensGenerated, generatedTokenIds, finishReason, false, generatedText.isEmpty())
                : diagnostics;
    }

    public SmolLM2RuntimeResult(String generatedText,
                                List<Integer> generatedTokenIds,
                                int tokensGenerated,
                                String finishReason) {
        this(generatedText, generatedTokenIds, tokensGenerated, finishReason, null);
    }
}
