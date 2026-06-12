package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGenerationLoop;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGenerationResult;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyStopTokenPolicy;

import java.util.List;
import java.util.Objects;
import java.util.function.IntConsumer;

/**
 * SmolLM2 adapter over the shared {@link DecoderOnlyGenerationLoop}.
 *
 * <p>Builds the SmolLM2 token sampler (stop policy + sampling/penalty from the request options), runs the shared loop
 * on the native WARP forward pass and maps the family-neutral {@link DecoderOnlyGenerationResult} onto SmolLM2's own
 * {@link SmolLM2TokenRuntimeResult}/{@link SmolLM2GenerationProfile}. The generation behaviour (sampling, stop policy,
 * KV cache, streaming, top-K diagnostics, timing split) lives in the shared loop and is unchanged.</p>
 */
final class SmolLM2WarpGenerationLoop {

    private static final int DEBUG_TOP_K = Integer.getInteger("smollm2.debug.topk", 0);
    private static final int DEBUG_STEPS = Integer.getInteger("smollm2.debug.steps", 3);

    private final SmolLM2TokenSamplerFactory tokenSamplerFactory;
    private final DecoderOnlyGenerationLoop loop;

    SmolLM2WarpGenerationLoop(SmolLM2WarpForwardPass forwardPass) {
        Objects.requireNonNull(forwardPass, "forwardPass");
        DecoderOnlyStopTokenPolicy stopPolicy = DecoderOnlyStopTokenPolicy.fromTokenIds(
                List.of(forwardPass.config().eosTokenId()));
        this.tokenSamplerFactory = new SmolLM2TokenSamplerFactory(stopPolicy);
        this.loop = new DecoderOnlyGenerationLoop(forwardPass.delegate(), DEBUG_TOP_K, DEBUG_STEPS);
    }

    SmolLM2TokenRuntimeResult generate(SmolLM2TokenRuntimeRequest request, IntConsumer generatedTokenConsumer) {
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
                SmolLM2ReferenceHotspotProfile.empty(),
                result.stepTopK(),
                result.decodeMicroProfile());
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
