package com.aresstack.windirectml.inference.decoderonly;

import java.util.List;

/**
 * Family-neutral result of one {@link DecoderOnlyGenerationLoop} run: the produced token ids plus the timing
 * breakdown and optional diagnostics. Model families map this onto their own runtime-result/profile types (adding
 * text-level timings such as tokenize/detokenize around the loop).
 *
 * <p>Result contract: {@code generatedTokenIds} / {@code fullTokenIds} contain only accepted user-output tokens; a
 * terminating stop token is NOT included (it ends generation but is not a generated token). When generation ended on a
 * stop token, {@code finishReason} is {@code "eos_token"} and {@code finishTokenId} carries that stop token id;
 * otherwise {@code finishTokenId} is {@code -1}.</p>
 */
public record DecoderOnlyGenerationResult(
        List<Integer> inputTokenIds,
        List<Integer> generatedTokenIds,
        List<Integer> fullTokenIds,
        int tokensGenerated,
        String finishReason,
        int finishTokenId,
        int maxNewTokens,
        long runtimeNanos,
        long prefillNanos,
        long decoderStepNanos,
        long lmHeadNanos,
        long tokenSelectNanos,
        List<String> stepTopK,
        List<String> decodeMicroProfile) {

    public DecoderOnlyGenerationResult {
        inputTokenIds = inputTokenIds == null ? List.of() : List.copyOf(inputTokenIds);
        generatedTokenIds = generatedTokenIds == null ? List.of() : List.copyOf(generatedTokenIds);
        fullTokenIds = fullTokenIds == null ? List.of() : List.copyOf(fullTokenIds);
        finishReason = finishReason == null ? "" : finishReason;
        stepTopK = stepTopK == null ? List.of() : List.copyOf(stepTopK);
        decodeMicroProfile = decodeMicroProfile == null ? List.of() : List.copyOf(decodeMicroProfile);
    }
}
