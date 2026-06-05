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
        SmolLM2ReferenceForwardProfile forwardProfile = new SmolLM2ReferenceForwardProfile();
        this.forwardPass = new SmolLM2ReferenceForwardPass(weights, forwardProfile);
        DecoderOnlyStopTokenPolicy stopPolicy = DecoderOnlyStopTokenPolicy.fromTokenIds(
                List.of(weights.config().eosTokenId()));
        this.tokenSamplerFactory = new SmolLM2TokenSamplerFactory(stopPolicy);
    }

    public SmolLM2TokenRuntimeResult generate(SmolLM2TokenRuntimeRequest request) {
        Objects.requireNonNull(request, "request");
        long runtimeStart = System.nanoTime();
        long prefillNanos = 0L;
        long decoderStepNanos = 0L;
        long tokenSelectNanos = 0L;
        List<Integer> fullTokenIds = new ArrayList<>(request.inputTokenIds());
        DecoderOnlyGeneratedTokens generatedTokens = new DecoderOnlyGeneratedTokens(request.maxNewTokens());
        String finishReason = "length";
        SmolLM2TokenSampler tokenSampler = tokenSamplerFactory.create(request.options());
        SmolLM2ReferenceKvCache kvCache = SmolLM2ReferenceKvCache.create(forwardPass.config());
        for (int i = 0; i < request.maxNewTokens(); i++) {
            long lmHeadBefore = forwardPass.profile().lmHeadNanos();
            long forwardStart = System.nanoTime();
            float[] logits = forwardPass.logitsForLastToken(fullTokenIds, kvCache);
            long forwardNanos = System.nanoTime() - forwardStart;
            long lmHeadDelta = Math.max(0L, forwardPass.profile().lmHeadNanos() - lmHeadBefore);
            long nonLmHeadForwardNanos = Math.max(0L, forwardNanos - lmHeadDelta);
            if (i == 0) {
                prefillNanos += nonLmHeadForwardNanos;
            } else {
                decoderStepNanos += nonLmHeadForwardNanos;
            }

            long tokenSelectStart = System.nanoTime();
            int nextToken = tokenSampler.selectNextToken(logits, generatedTokens);
            tokenSelectNanos += System.nanoTime() - tokenSelectStart;

            generatedTokens.add(nextToken);
            fullTokenIds.add(nextToken);
            if (tokenSampler.shouldStop(nextToken)) {
                finishReason = "eos_token";
                break;
            }
        }
        SmolLM2GenerationProfile profile = new SmolLM2GenerationProfile(
                System.nanoTime() - runtimeStart,
                0L,
                prefillNanos,
                decoderStepNanos,
                forwardPass.profile().lmHeadNanos(),
                tokenSelectNanos,
                0L);
        return new SmolLM2TokenRuntimeResult(
                request.inputTokenIds(),
                toList(generatedTokens),
                fullTokenIds,
                generatedTokens.count(),
                finishReason,
                request.maxNewTokens(),
                profile);
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
