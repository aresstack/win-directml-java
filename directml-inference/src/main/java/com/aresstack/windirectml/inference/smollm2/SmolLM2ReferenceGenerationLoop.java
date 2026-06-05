package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGeneratedTokens;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyStopTokenPolicy;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Token-level SmolLM2 reference generator.
 */
public final class SmolLM2ReferenceGenerationLoop {

    private final SmolLM2ReferenceForwardPass forwardPass;
    private final SmolLM2TokenSamplerFactory tokenSamplerFactory;

    public SmolLM2ReferenceGenerationLoop(SmolLM2Weights weights) {
        Objects.requireNonNull(weights, "weights");
        this.forwardPass = new SmolLM2ReferenceForwardPass(weights);
        DecoderOnlyStopTokenPolicy stopPolicy = DecoderOnlyStopTokenPolicy.fromTokenIds(
                List.of(weights.config().eosTokenId()));
        this.tokenSamplerFactory = new SmolLM2TokenSamplerFactory(stopPolicy);
    }

    public SmolLM2TokenRuntimeResult generate(SmolLM2TokenRuntimeRequest request) {
        Objects.requireNonNull(request, "request");
        List<Integer> fullTokenIds = new ArrayList<>(request.inputTokenIds());
        DecoderOnlyGeneratedTokens generatedTokens = new DecoderOnlyGeneratedTokens(request.maxNewTokens());
        String finishReason = "length";
        SmolLM2TokenSampler tokenSampler = tokenSamplerFactory.create(request.options());
        SmolLM2ReferenceKvCache kvCache = SmolLM2ReferenceKvCache.create(forwardPass.config());
        for (int i = 0; i < request.maxNewTokens(); i++) {
            float[] logits = forwardPass.logitsForLastToken(fullTokenIds, kvCache);
            int nextToken = tokenSampler.selectNextToken(logits, generatedTokens);
            generatedTokens.add(nextToken);
            fullTokenIds.add(nextToken);
            if (tokenSampler.shouldStop(nextToken)) {
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


    private static List<Integer> toList(DecoderOnlyGeneratedTokens tokens) {
        int[] ids = tokens.copyTokenIds();
        List<Integer> values = new ArrayList<>(ids.length);
        for (int id : ids) {
            values.add(id);
        }
        return values;
    }
}
