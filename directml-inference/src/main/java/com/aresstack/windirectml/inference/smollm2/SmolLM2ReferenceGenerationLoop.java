package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGeneratedTokens;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyStopTokenPolicy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.IntConsumer;

/**
 * Token-level SmolLM2 reference generator.
 */
public final class SmolLM2ReferenceGenerationLoop {

    private static final Logger log = LoggerFactory.getLogger(SmolLM2ReferenceGenerationLoop.class);

    /**
     * Opt-in numerical diagnostic. Launch with {@code -Dsmollm2.debug.topk=10} to log the
     * top-K raw logits (before any repetition penalty) and the greedy pick for the first
     * {@code -Dsmollm2.debug.steps} decode steps (default 3). This is the seam for comparing
     * SmolLM2's CPU reference forward pass against a Hugging Face Transformers reference run
     * on identical input token IDs.
     */
    private static final int DEBUG_TOP_K = Integer.getInteger("smollm2.debug.topk", 0);
    private static final int DEBUG_STEPS = Integer.getInteger("smollm2.debug.steps", 3);

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
        return generate(request, null);
    }

    public SmolLM2TokenRuntimeResult generate(SmolLM2TokenRuntimeRequest request, IntConsumer generatedTokenConsumer) {
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
        int topKSteps = Math.max(DEBUG_STEPS, 3);
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

            // Always capture the first few steps' top-K raw logits (pre-penalty) so the
            // numerical comparison against a Hugging Face Transformers reference is visible
            // in the workbench output without any VM flag.
            if (i < topKSteps) {
                String line = topKLine(i, logits, 10);
                stepTopK.add(line);
                if (DEBUG_TOP_K > 0 && i < DEBUG_STEPS) {
                    log.info(line);
                }
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
                forwardPass.profile().lmHeadNanos(),
                tokenSelectNanos,
                0L,
                forwardPass.profile().snapshot(),
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

    /**
     * Build a "top-{@code k} raw logits (pre-penalty)" line for one step, listing token IDs
     * and logit values. This is the seam for a numerical diff against a Hugging Face
     * Transformers reference run on identical input token IDs.
     */
    private static String topKLine(int step, float[] logits, int k) {
        int topK = Math.min(k, logits.length);
        int[] bestIds = new int[topK];
        float[] bestVals = new float[topK];
        java.util.Arrays.fill(bestIds, -1);
        java.util.Arrays.fill(bestVals, Float.NEGATIVE_INFINITY);
        for (int id = 0; id < logits.length; id++) {
            float value = logits[id];
            // Insert into the descending-sorted top-K buffer.
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
        sb.append("top-").append(topK).append(" @ step ").append(step).append(" (raw, pre-penalty): ");
        for (int rank = 0; rank < topK; rank++) {
            if (rank > 0) {
                sb.append(", ");
            }
            sb.append('#').append(bestIds[rank]).append('=').append(String.format("%.4f", bestVals[rank]));
        }
        return sb.toString();
    }
}
