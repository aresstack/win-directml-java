package com.aresstack.windirectml.inference.t5;

import java.util.Arrays;
import java.util.Objects;

/**
 * T5 generation loop.
 *
 * <p>The loop owns the seq2seq orchestration and delegates vocabulary projection
 * to a {@link T5LogitProjector}. This lets the reference path and the first
 * WARP-backed LM-head bridge share one token loop without copying decoder-only
 * runtime logic.</p>
 */
public final class T5GenerationLoop {
    private final T5EncoderRunner encoderRunner;
    private final T5DecoderRunner decoderRunner;
    private final T5LogitProjector logitProjector;
    private final T5TokenSelector tokenSelector;

    private T5GenerationLoop(T5EncoderRunner encoderRunner,
                             T5DecoderRunner decoderRunner,
                             T5LogitProjector logitProjector,
                             T5TokenSelector tokenSelector) {
        this.encoderRunner = Objects.requireNonNull(encoderRunner, "encoderRunner");
        this.decoderRunner = Objects.requireNonNull(decoderRunner, "decoderRunner");
        this.logitProjector = Objects.requireNonNull(logitProjector, "logitProjector");
        this.tokenSelector = Objects.requireNonNull(tokenSelector, "tokenSelector");
    }

    public static T5GenerationLoop greedy(T5EncoderRunner encoderRunner,
                                          T5DecoderRunner decoderRunner,
                                          T5Weights weights) {
        return greedy(encoderRunner, decoderRunner, T5LmHead.from(weights));
    }

    public static T5GenerationLoop greedy(T5EncoderRunner encoderRunner,
                                          T5DecoderRunner decoderRunner,
                                          T5LogitProjector logitProjector) {
        return new T5GenerationLoop(encoderRunner, decoderRunner, logitProjector, T5TokenSelector.greedy());
    }

    public T5RuntimeResult generate(T5RuntimeRequest request) {
        Objects.requireNonNull(request, "request");
        requireGreedyRequest(request);
        T5GenerationMetricsCollector metrics = new T5GenerationMetricsCollector();
        long runtimeStart = System.nanoTime();
        try (T5GenerationProfiler.Scope ignored = T5GenerationProfiler.open(metrics)) {
            long encoderStart = System.nanoTime();
            T5EncoderOutput encoderOutput = encoderRunner.encode(request.inputTokenIds());
            metrics.addEncoderNanos(System.nanoTime() - encoderStart);
            int[] generated = new int[request.maxNewTokens()];
            int generatedTokens = 0;
            int decoderTokenId = request.decoderStartTokenId();
            T5DecoderCache cache = T5DecoderCache.empty();
            for (int step = 0; step < request.maxNewTokens(); step++) {
                long decodeStart = System.nanoTime();
                T5DecoderState state = decoderRunner.decodeStep(decoderTokenId, encoderOutput, cache);
                metrics.addDecodeNanos(System.nanoTime() - decodeStart);

                long lmHeadStart = System.nanoTime();
                float[] logits = logitProjector.logits(state.lastHiddenState());
                metrics.addLmHeadNanos(System.nanoTime() - lmHeadStart);

                long tokenSelectionStart = System.nanoTime();
                suppressControlTokens(logits, request, step);
                int nextTokenId = tokenSelector.select(logits);
                metrics.addTokenSelectionNanos(System.nanoTime() - tokenSelectionStart);

                generated[generatedTokens] = nextTokenId;
                generatedTokens++;
                cache = cache.append(decoderTokenId, state);
                if (request.stopTokenPolicy().shouldStop(nextTokenId)) {
                    return result(generated, generatedTokens, T5RuntimeResult.FinishReason.stop_token,
                            metrics, runtimeStart);
                }
                decoderTokenId = nextTokenId;
            }
            return result(generated, generatedTokens, T5RuntimeResult.FinishReason.max_tokens, metrics, runtimeStart);
        }
    }

    private static T5RuntimeResult result(int[] generated, int generatedTokens,
                                          T5RuntimeResult.FinishReason finishReason,
                                          T5GenerationMetricsCollector metrics, long runtimeStart) {
        T5GenerationMetrics generationMetrics = metrics.finish(System.nanoTime() - runtimeStart, generatedTokens);
        return new T5RuntimeResult(Arrays.copyOf(generated, generatedTokens),
                finishReason, generatedTokens, generationMetrics);
    }

    private static void suppressControlTokens(float[] logits, T5RuntimeRequest request, int step) {
        for (Integer tokenId : request.suppressedTokenIds()) {
            suppressToken(logits, tokenId);
        }
        if (step < request.minimumTokensBeforeStop()) {
            for (int token = 0; token < logits.length; token++) {
                if (request.stopTokenPolicy().shouldStop(token)) {
                    suppressToken(logits, token);
                }
            }
        }
    }

    private static void suppressToken(float[] logits, int tokenId) {
        if (tokenId >= 0 && tokenId < logits.length) {
            logits[tokenId] = -Float.MAX_VALUE;
        }
    }

    private static void requireGreedyRequest(T5RuntimeRequest request) {
        if (request.temperature() != 0.0f) {
            throw new IllegalArgumentException("T5 reference generation supports greedy decoding only: temperature="
                    + request.temperature());
        }
        if (request.topK() != 0) {
            throw new IllegalArgumentException("T5 reference generation supports greedy decoding only: topK="
                    + request.topK());
        }
    }
}
