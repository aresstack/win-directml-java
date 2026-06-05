package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGeneratedTokens;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGreedyTokenSelector;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyStopTokenPolicy;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Token-level SmolLM2 reference generator.
 */
public final class SmolLM2ReferenceGenerationLoop {

    private final SmolLM2ReferenceForwardPass forwardPass;
    private final DecoderOnlyGreedyTokenSelector tokenSelector;

    public SmolLM2ReferenceGenerationLoop(SmolLM2Weights weights) {
        Objects.requireNonNull(weights, "weights");
        this.forwardPass = new SmolLM2ReferenceForwardPass(weights);
        DecoderOnlyStopTokenPolicy stopPolicy = DecoderOnlyStopTokenPolicy.fromTokenIds(
                List.of(weights.config().eosTokenId()));
        this.tokenSelector = new DecoderOnlyGreedyTokenSelector(stopPolicy);
    }

    public SmolLM2TokenRuntimeResult generate(SmolLM2TokenRuntimeRequest request) {
        Objects.requireNonNull(request, "request");
        List<Integer> fullTokenIds = new ArrayList<>(request.inputTokenIds());
        DecoderOnlyGeneratedTokens generatedTokens = new DecoderOnlyGeneratedTokens(request.maxNewTokens());
        String finishReason = "length";
        float repetitionPenalty = repetitionPenalty(request.options());
        for (int i = 0; i < request.maxNewTokens(); i++) {
            float[] logits = forwardPass.logitsForLastToken(fullTokenIds);
            int nextToken = tokenSelector.selectNextToken(logits, generatedTokens, repetitionPenalty);
            generatedTokens.add(nextToken);
            fullTokenIds.add(nextToken);
            if (tokenSelector.shouldStop(nextToken)) {
                finishReason = "eos_token";
                break;
            }
        }
        return new SmolLM2TokenRuntimeResult(
                request.inputTokenIds(),
                toList(generatedTokens),
                fullTokenIds,
                generatedTokens.count(),
                finishReason);
    }

    private static float repetitionPenalty(SmolLM2GenerationOptions options) {
        // Keep the public options type small for now. Add an explicit field later if sampling is implemented.
        return 1.0f;
    }

    private static List<Integer> toList(DecoderOnlyGeneratedTokens tokens) {
        int[] ids = tokens.copyTokenIds();
        List<Integer> values = new ArrayList<>(ids.length);
        for (int id : ids) {
            values.add(id);
        }
        return values;
    }
}
