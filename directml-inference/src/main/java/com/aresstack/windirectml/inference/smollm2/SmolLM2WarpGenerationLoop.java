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
 * Token-level SmolLM2 generator running on the native {@link SmolLM2WarpForwardPass}.
 *
 * <p>Mirrors {@link SmolLM2ReferenceGenerationLoop} one-to-one (sampling, stop policy, KV cache, top-K diagnostics)
 * but feeds logits from the WARP forward pass instead of the CPU reference forward pass. Keeping the loops aligned
 * makes the numerical WARP-vs-reference comparison a pure forward-pass diff.</p>
 */
final class SmolLM2WarpGenerationLoop {

    private static final Logger log = LoggerFactory.getLogger(SmolLM2WarpGenerationLoop.class);

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
        long lmHeadNanos = 0L;
        long tokenSelectNanos = 0L;
        SmolLM2WarpDecodeProfile decodeProfile = forwardPass.decodeProfile();
        boolean profileDecode = decodeProfile.enabled();
        if (profileDecode) {
            decodeProfile.reset();
        }
        List<Integer> fullTokenIds = new ArrayList<>(request.inputTokenIds());
        DecoderOnlyGeneratedTokens generatedTokens = new DecoderOnlyGeneratedTokens(request.maxNewTokens());
        String finishReason = "length";
        SmolLM2TokenSampler tokenSampler = tokenSamplerFactory.create(request.options());
        int maxTokens = request.inputTokenIds().size() + request.maxNewTokens();
        SmolLM2WarpKvCache kvCache = SmolLM2WarpKvCache.create(forwardPass.config(), maxTokens);
        List<String> stepTopK = new ArrayList<>();
        for (int i = 0; i < request.maxNewTokens(); i++) {
            long forwardStart = System.nanoTime();
            float[] logits = forwardPass.logitsForLastToken(fullTokenIds, kvCache);
            long forwardNanos = System.nanoTime() - forwardStart;
            // Report the LM-head projection separately instead of silently folding it into prefill/decoder.
            long stepLmHeadNanos = Math.min(forwardPass.lastCallLmHeadNanos(), forwardNanos);
            lmHeadNanos += stepLmHeadNanos;
            long computeNanos = forwardNanos - stepLmHeadNanos;
            if (i == 0) {
                prefillNanos += computeNanos;
            } else {
                decoderStepNanos += computeNanos;
            }

            if (DEBUG_TOP_K > 0 && i < DEBUG_STEPS) {
                stepTopK.add(topKLine(i, logits, DEBUG_TOP_K));
            }

            long tokenSelectStart = System.nanoTime();
            int nextToken = tokenSampler.selectNextToken(logits, generatedTokens);
            long tokenSelectStep = System.nanoTime() - tokenSelectStart;
            tokenSelectNanos += tokenSelectStep;
            if (profileDecode) {
                decodeProfile.tokenSelect += tokenSelectStep;
            }

            generatedTokens.add(nextToken);
            fullTokenIds.add(nextToken);
            if (generatedTokenConsumer != null && !tokenSampler.shouldStop(nextToken)) {
                long streamStart = System.nanoTime();
                generatedTokenConsumer.accept(nextToken);
                if (profileDecode) {
                    decodeProfile.streamingCallback += System.nanoTime() - streamStart;
                }
            }
            if (tokenSampler.shouldStop(nextToken)) {
                finishReason = "eos_token";
                break;
            }
        }
        List<String> microProfileLines = List.of();
        if (profileDecode) {
            microProfileLines = decodeProfile.format();
            if (log.isInfoEnabled()) {
                for (String line : microProfileLines) {
                    log.info(line);
                }
            }
        }
        SmolLM2GenerationProfile profile = new SmolLM2GenerationProfile(
                System.nanoTime() - runtimeStart,
                0L,
                prefillNanos,
                decoderStepNanos,
                lmHeadNanos,
                tokenSelectNanos,
                0L,
                SmolLM2ReferenceHotspotProfile.empty(),
                stepTopK,
                microProfileLines);
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
