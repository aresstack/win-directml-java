package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGenerationLoop;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGenerationResult;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyStopTokenPolicy;

import java.util.List;
import java.util.Objects;
import java.util.function.IntConsumer;

/**
 * SmolLM2 CPU reference generator — a thin adapter over the shared {@link DecoderOnlyGenerationLoop} (todo item 4).
 *
 * <p>It no longer contains its own decode loop: it builds the SmolLM2 token sampler (shared
 * {@link DecoderOnlyStopTokenPolicy} + sampling/penalty from the request options), drives the shared loop on the
 * reference forward pass via {@link SmolLM2ReferenceDecoderOnlyForwardPass}, and maps the family-neutral
 * {@link DecoderOnlyGenerationResult} onto SmolLM2's {@link SmolLM2TokenRuntimeResult}. Behaviour (generated token
 * sequence, finish reason, streaming order, prefill/decode/LM-head timing split, top-K diagnostics) is identical to
 * the WARP path because both now run the same loop.</p>
 */
public final class SmolLM2ReferenceGenerationLoop {

    /** Opt-in top-K logit diagnostic; same switches as the WARP path. */
    private static final int DEBUG_TOP_K = Integer.getInteger("smollm2.debug.topk", 0);
    private static final int DEBUG_STEPS = Integer.getInteger("smollm2.debug.steps", 3);

    private final SmolLM2ReferenceForwardProfile forwardProfile;
    private final SmolLM2TokenSamplerFactory tokenSamplerFactory;
    private final DecoderOnlyGenerationLoop loop;

    public SmolLM2ReferenceGenerationLoop(SmolLM2Weights weights) {
        Objects.requireNonNull(weights, "weights");
        this.forwardProfile = new SmolLM2ReferenceForwardProfile();
        SmolLM2ReferenceForwardPass forwardPass = new SmolLM2ReferenceForwardPass(weights, forwardProfile);
        DecoderOnlyStopTokenPolicy stopPolicy = DecoderOnlyStopTokenPolicy.fromTokenIds(
                List.of(weights.config().eosTokenId()));
        this.tokenSamplerFactory = new SmolLM2TokenSamplerFactory(stopPolicy);
        this.loop = new DecoderOnlyGenerationLoop(
                new SmolLM2ReferenceDecoderOnlyForwardPass(forwardPass), DEBUG_TOP_K, DEBUG_STEPS);
    }

    public SmolLM2TokenRuntimeResult generate(SmolLM2TokenRuntimeRequest request) {
        return generate(request, null);
    }

    public SmolLM2TokenRuntimeResult generate(SmolLM2TokenRuntimeRequest request, IntConsumer generatedTokenConsumer) {
        Objects.requireNonNull(request, "request");
        SmolLM2TokenSampler tokenSampler = tokenSamplerFactory.create(request.options());
        DecoderOnlyGenerationResult result = loop.generate(
                request.inputTokenIds(), request.maxNewTokens(), tokenSampler, generatedTokenConsumer);

        SmolLM2GenerationProfile profile = new SmolLM2GenerationProfile(
                result.runtimeNanos(),
                0L,
                result.prefillNanos(),
                result.decoderStepNanos(),
                result.lmHeadNanos(),
                result.tokenSelectNanos(),
                0L,
                forwardProfile.snapshot(),
                result.stepTopK());
        return new SmolLM2TokenRuntimeResult(
                result.inputTokenIds(),
                result.generatedTokenIds(),
                result.fullTokenIds(),
                result.tokensGenerated(),
                result.finishReason(),
                result.maxNewTokens(),
                profile);
    }
}
