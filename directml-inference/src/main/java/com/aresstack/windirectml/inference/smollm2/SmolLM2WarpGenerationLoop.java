package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGeneratedTokens;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyStopTokenPolicy;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.IntConsumer;

/**
 * Token-level SmolLM2 generator running on the native {@link SmolLM2WarpForwardPass}.
 *
 * <p>Mirrors {@link SmolLM2ReferenceGenerationLoop} one-to-one (sampling, stop policy, KV cache, top-K diagnostics)
 * but feeds logits from the WARP forward pass instead of the CPU reference forward pass. Keeping the loops aligned
 * makes the numerical WARP-vs-reference comparison a pure forward-pass diff.</p>
 */
final class SmolLM2WarpGenerationLoop {

    private static final int DEBUG_TOP_K = Integer.getInteger("smollm2.debug.topk", 0);
    private static final int DEBUG_STEPS = Integer.getInteger("smollm2.debug.steps", 3);

    private final SmolLM2WarpForwardPass forwardPass;
    private final SmolLM2TokenSamplerFactory tokenSamplerFactory;

    SmolLM2WarpGenerationLoop(SmolLM2WarpForwardPass forwardPass) {
        this.forwardPass = Objects.requireNonNull(forwardPass, "forwardPass");
        DecoderOnlyStopTokenPolicy stopPolicy = DecoderOnlyStopTokenPolicy.fromTokenIds(
                List.of(forwardPass.config().eosTokenId()));
        this.tokenSamplerFactory = new SmolLM2TokenSamplerFactory(stopPolicy);
    }

    SmolLM2TokenRuntimeResult generate(SmolLM2TokenRuntimeRequest request, IntConsumer generatedTokenConsumer) {
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
        List<String> stepTopK = new ArrayList<>();
        for (int i = 0; i < request.maxNewTokens(); i++) {
            long forwardStart = System.nanoTime();
            float[] logits = forwardPass.logitsForLastToken(fullTokenIds, kvCache);
            long forwardNanos = System.nanoTime() - forwardStart;
            if (i == 0) {
                prefillNanos += forwardNanos;
            } else {
                decoderStepNanos += forwardNanos;
            }

            if (DEBUG_TOP_K > 0 && i < DEBUG_STEPS) {
                stepTopK.add(topKLine(i, logits, DEBUG_TOP_K));
            }

            long tokenSelectStart = System.nanoTime();
            int nextToken = tokenSampler.selectNextToken(logits, generatedTokens);
            tokenSelectNanos += System.nanoTime() - tokenSelectStart;

            generatedTokens.add(nextToken);
            fullTokenIds.add(nextToken);
            if (generatedTokenConsumer != null && !tokenSampler.shouldStop(nextToken)) {
                generatedTokenConsumer.accept(nextToken);
            }
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
                0L,
                tokenSelectNanos,
                0L,
                SmolLM2ReferenceHotspotProfile.empty(),
                stepTopK);
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

    private static String topKLine(int step, float[] logits, int k) {
        int topK = Math.min(k, logits.length);
        int[] bestIds = new int[topK];
        float[] bestVals = new float[topK];
        java.util.Arrays.fill(bestIds, -1);
        java.util.Arrays.fill(bestVals, Float.NEGATIVE_INFINITY);
        for (int id = 0; id < logits.length; id++) {
            float value = logits[id];
            if (value <= bestVals[topK - 1]) {
                continue;
            }
            int pos = topK - 1;
            while (pos > 0 && bestVals[pos - 1] < value) {
                bestVals[pos] = bestVals[pos - 1];
                bestIds[pos] = bestIds[pos - 1];
                pos--;
            }
            bestVals[pos] = value;
            bestIds[pos] = id;
        }
        StringBuilder sb = new StringBuilder(64 + topK * 24);
        sb.append("warp top-").append(topK).append(" @ step ").append(step).append(" (raw, pre-penalty): ");
        for (int rank = 0; rank < topK; rank++) {
            if (rank > 0) {
                sb.append(", ");
            }
            sb.append('#').append(bestIds[rank]).append('=').append(String.format("%.4f", bestVals[rank]));
        }
        return sb.toString();
    }
}
