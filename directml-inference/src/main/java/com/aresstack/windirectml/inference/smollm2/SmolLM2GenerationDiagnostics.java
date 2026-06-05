package com.aresstack.windirectml.inference.smollm2;

import java.util.List;

/**
 * Diagnostic metadata produced by the SmolLM2 reference generation path.
 */
public record SmolLM2GenerationDiagnostics(int inputTokenCount,
                                           int outputTokenCount,
                                           int fullTokenCount,
                                           int maxNewTokens,
                                           List<Integer> generatedTokenIds,
                                           String finishReason,
                                           boolean immediateEos,
                                           boolean emptyDecodedOutput,
                                           SmolLM2GenerationProfile profile) {

    public SmolLM2GenerationDiagnostics {
        if (inputTokenCount < 0) {
            throw new IllegalArgumentException("inputTokenCount must not be negative");
        }
        if (outputTokenCount < 0) {
            throw new IllegalArgumentException("outputTokenCount must not be negative");
        }
        if (fullTokenCount < 0) {
            throw new IllegalArgumentException("fullTokenCount must not be negative");
        }
        if (maxNewTokens < 0) {
            throw new IllegalArgumentException("maxNewTokens must not be negative");
        }

        generatedTokenIds = generatedTokenIds == null ? List.of() : List.copyOf(generatedTokenIds);
        finishReason = finishReason == null ? "" : finishReason;
        profile = profile == null ? SmolLM2GenerationProfile.empty() : profile;
    }

    public SmolLM2GenerationDiagnostics(int inputTokenCount,
                                         int outputTokenCount,
                                         int fullTokenCount,
                                         int maxNewTokens,
                                         List<Integer> generatedTokenIds,
                                         String finishReason,
                                         boolean immediateEos,
                                         boolean emptyDecodedOutput) {
        this(inputTokenCount, outputTokenCount, fullTokenCount, maxNewTokens, generatedTokenIds, finishReason,
                immediateEos, emptyDecodedOutput, SmolLM2GenerationProfile.empty());
    }

    /**
     * Create diagnostics from the token generation result and decoded visible text.
     */
    public static SmolLM2GenerationDiagnostics fromTokenResult(SmolLM2TokenRuntimeResult result,
                                                               String generatedText) {
        if (result == null) {
            return empty(generatedText);
        }
        return fromTokenResult(result, generatedText, result.profile());
    }

    /**
     * Create diagnostics from the token generation result, decoded visible text and explicit timing profile.
     */
    public static SmolLM2GenerationDiagnostics fromTokenResult(SmolLM2TokenRuntimeResult result,
                                                               String generatedText,
                                                               SmolLM2GenerationProfile profile) {
        if (result == null) {
            return empty(generatedText);
        }

        List<Integer> inputTokenIds = result.inputTokenIds() == null ? List.of() : result.inputTokenIds();
        List<Integer> generatedTokenIds = result.generatedTokenIds() == null ? List.of() : result.generatedTokenIds();
        List<Integer> fullTokenIds = result.fullTokenIds() == null ? List.of() : result.fullTokenIds();

        String finishReason = result.finishReason() == null ? "" : result.finishReason();
        boolean immediateEos = result.tokensGenerated() == 1 && "eos_token".equals(finishReason);
        boolean emptyDecodedOutput = generatedText == null || generatedText.isEmpty();

        return new SmolLM2GenerationDiagnostics(
                inputTokenIds.size(),
                result.tokensGenerated(),
                fullTokenIds.size(),
                result.maxNewTokens(),
                generatedTokenIds,
                finishReason,
                immediateEos,
                emptyDecodedOutput,
                profile);
    }

    /**
     * Return a compact preview for diagnostic log output.
     */
    public String generatedTokenIdsPreview(int limit) {
        if (limit <= 0 || generatedTokenIds.isEmpty()) {
            return "[]";
        }

        int visibleCount = Math.min(limit, generatedTokenIds.size());
        StringBuilder builder = new StringBuilder("[");
        for (int i = 0; i < visibleCount; i++) {
            if (i > 0) {
                builder.append(", ");
            }
            builder.append(generatedTokenIds.get(i));
        }

        if (visibleCount < generatedTokenIds.size()) {
            builder.append(", ...");
        }

        return builder.append(']').toString();
    }

    private static SmolLM2GenerationDiagnostics empty(String generatedText) {
        return new SmolLM2GenerationDiagnostics(
                0,
                0,
                0,
                0,
                List.of(),
                "",
                false,
                generatedText == null || generatedText.isEmpty(),
                SmolLM2GenerationProfile.empty());
    }
}
