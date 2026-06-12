package com.aresstack.windirectml.inference.decoderonly;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.IntConsumer;

/**
 * Family-neutral token generation loop for decoder-only WARP runtimes.
 *
 * <p>Drives a per-run {@link DecoderOnlyDecodeSession} (obtained from the {@link DecoderOnlyForwardPass}) step by
 * step: prefill → next-token selection → optional streaming → stop check → decodeNext, with per-stage timing. The
 * session owns its KV cache, so the loop never creates or passes one. The numerical decisions
 * (sampling, repetition penalty, stop tokens) live entirely in the supplied {@link DecoderOnlyTokenSelector}, so this
 * loop is identical for every decoder-only family; only the selector and the result mapping differ.</p>
 *
 * <p>Extracted verbatim from the previous SmolLM2-specific loop; behaviour (token sequence, finish reason, streaming
 * order, timing split, top-K diagnostics) is unchanged.</p>
 */
public final class DecoderOnlyGenerationLoop {

    private static final Logger log = LoggerFactory.getLogger(DecoderOnlyGenerationLoop.class);

    private final DecoderOnlyForwardPass forwardPass;
    private final int debugTopK;
    private final int debugSteps;

    /**
     * @param forwardPass the logits source
     * @param debugTopK   if &gt; 0, record a top-K diagnostic line for the first {@code debugSteps} steps
     * @param debugSteps  number of initial steps to record top-K diagnostics for
     */
    public DecoderOnlyGenerationLoop(DecoderOnlyForwardPass forwardPass, int debugTopK, int debugSteps) {
        this.forwardPass = Objects.requireNonNull(forwardPass, "forwardPass");
        this.debugTopK = debugTopK;
        this.debugSteps = debugSteps;
    }

    /**
     * Generate up to {@code maxNewTokens} tokens after {@code inputTokenIds}.
     *
     * @param inputTokenIds          the prompt token ids
     * @param maxNewTokens           maximum number of tokens to generate
     * @param selector               next-token selection + stop policy
     * @param generatedTokenConsumer optional sink invoked for every accepted (non-stop) token, in order
     */
    public DecoderOnlyGenerationResult generate(List<Integer> inputTokenIds,
                                                int maxNewTokens,
                                                DecoderOnlyTokenSelector selector,
                                                IntConsumer generatedTokenConsumer) {
        Objects.requireNonNull(inputTokenIds, "inputTokenIds");
        Objects.requireNonNull(selector, "selector");
        long runtimeStart = System.nanoTime();
        long prefillNanos = 0L;
        long decoderStepNanos = 0L;
        long lmHeadNanos = 0L;
        long tokenSelectNanos = 0L;
        DecoderOnlyWarpDecodeProfile decodeProfile = forwardPass.decodeProfile();
        boolean profileDecode = decodeProfile.enabled();
        if (profileDecode) {
            decodeProfile.reset();
        }
        List<Integer> fullTokenIds = new ArrayList<>(inputTokenIds);
        DecoderOnlyGeneratedTokens generatedTokens = new DecoderOnlyGeneratedTokens(maxNewTokens);
        String finishReason = "length";
        int finishTokenId = -1;
        int maxTokens = inputTokenIds.size() + maxNewTokens;
        List<String> stepTopK = new ArrayList<>();
        // The session owns its decode state / KV cache; the loop never creates or passes a cache. Step 0 prefills the
        // prompt; every later step feeds the token selected in the previous step. This is the same processing as the
        // earlier cache-in-parameter loop (prefill of the whole prompt, then one new token per decode step).
        try (DecoderOnlyDecodeSession session = forwardPass.newDecodeSession(maxTokens)) {
            int pendingToken = -1;
            for (int i = 0; i < maxNewTokens; i++) {
                long forwardStart = System.nanoTime();
                float[] logits = (i == 0) ? session.prefill(inputTokenIds) : session.decodeNext(pendingToken);
                long forwardNanos = System.nanoTime() - forwardStart;
                // Report the LM-head projection separately instead of silently folding it into prefill/decoder.
                long stepLmHeadNanos = Math.min(session.lastCallLmHeadNanos(), forwardNanos);
                lmHeadNanos += stepLmHeadNanos;
                long computeNanos = forwardNanos - stepLmHeadNanos;
                if (i == 0) {
                    prefillNanos += computeNanos;
                } else {
                    decoderStepNanos += computeNanos;
                }

                if (debugTopK > 0 && i < debugSteps) {
                    stepTopK.add(topKLine(i, logits, debugTopK));
                }

                long tokenSelectStart = System.nanoTime();
                int nextToken = selector.selectNextToken(logits, generatedTokens);
                long tokenSelectStep = System.nanoTime() - tokenSelectStart;
                tokenSelectNanos += tokenSelectStep;
                if (profileDecode) {
                    decodeProfile.tokenSelect += tokenSelectStep;
                }

                // Harmonised result contract: a terminating stop token ends generation but is NOT recorded as a
                // generated/streamed user-output token (matches the production Qwen runtime). It is reported via
                // finishTokenId instead.
                if (selector.shouldStop(nextToken)) {
                    finishReason = "eos_token";
                    finishTokenId = nextToken;
                    break;
                }
                generatedTokens.add(nextToken);
                fullTokenIds.add(nextToken);
                pendingToken = nextToken;
                if (generatedTokenConsumer != null) {
                    long streamStart = System.nanoTime();
                    generatedTokenConsumer.accept(nextToken);
                    if (profileDecode) {
                        decodeProfile.streamingCallback += System.nanoTime() - streamStart;
                    }
                }
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
        return new DecoderOnlyGenerationResult(
                inputTokenIds,
                toList(generatedTokens),
                fullTokenIds,
                generatedTokens.count(),
                finishReason,
                finishTokenId,
                maxNewTokens,
                System.nanoTime() - runtimeStart,
                prefillNanos,
                decoderStepNanos,
                lmHeadNanos,
                tokenSelectNanos,
                stepTopK,
                microProfileLines);
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
        Arrays.fill(bestIds, -1);
        Arrays.fill(bestVals, Float.NEGATIVE_INFINITY);
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
