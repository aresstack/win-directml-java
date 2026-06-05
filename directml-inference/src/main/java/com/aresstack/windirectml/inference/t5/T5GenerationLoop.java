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
        T5EncoderOutput encoderOutput = encoderRunner.encode(request.inputTokenIds());
        int[] generated = new int[request.maxNewTokens()];
        int generatedTokens = 0;
        int decoderTokenId = request.decoderStartTokenId();
        T5DecoderCache cache = T5DecoderCache.empty();
        for (int step = 0; step < request.maxNewTokens(); step++) {
            T5DecoderState state = decoderRunner.decodeStep(decoderTokenId, encoderOutput, cache);
            float[] logits = logitProjector.logits(state.lastHiddenState());
            int nextTokenId = tokenSelector.select(logits);
            generated[generatedTokens] = nextTokenId;
            generatedTokens++;
            cache = cache.append(decoderTokenId, state);
            if (request.stopTokenPolicy().shouldStop(nextTokenId)) {
                return new T5RuntimeResult(Arrays.copyOf(generated, generatedTokens),
                        T5RuntimeResult.FinishReason.stop_token, generatedTokens);
            }
            decoderTokenId = nextTokenId;
        }
        return new T5RuntimeResult(Arrays.copyOf(generated, generatedTokens),
                T5RuntimeResult.FinishReason.max_tokens, generatedTokens);
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
