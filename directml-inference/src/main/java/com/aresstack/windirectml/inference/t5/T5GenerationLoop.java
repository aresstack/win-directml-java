package com.aresstack.windirectml.inference.t5;

import java.util.Arrays;
import java.util.Objects;

/**
 * Reference T5 generation loop for v38.
 *
 * <p>This loop validates the encoder/decoder/LM-head orchestration before WARP
 * kernels replace the Java reference math.</p>
 */
public final class T5GenerationLoop {
    private final T5EncoderPipeline encoderPipeline;
    private final T5DecoderPipeline decoderPipeline;
    private final T5LmHead lmHead;
    private final T5TokenSelector tokenSelector;

    private T5GenerationLoop(T5EncoderPipeline encoderPipeline,
                             T5DecoderPipeline decoderPipeline,
                             T5LmHead lmHead,
                             T5TokenSelector tokenSelector) {
        this.encoderPipeline = Objects.requireNonNull(encoderPipeline, "encoderPipeline");
        this.decoderPipeline = Objects.requireNonNull(decoderPipeline, "decoderPipeline");
        this.lmHead = Objects.requireNonNull(lmHead, "lmHead");
        this.tokenSelector = Objects.requireNonNull(tokenSelector, "tokenSelector");
    }

    public static T5GenerationLoop greedy(T5EncoderPipeline encoderPipeline,
                                          T5DecoderPipeline decoderPipeline,
                                          T5Weights weights) {
        return new T5GenerationLoop(encoderPipeline, decoderPipeline, T5LmHead.from(weights), T5TokenSelector.greedy());
    }

    public T5RuntimeResult generate(T5RuntimeRequest request) {
        Objects.requireNonNull(request, "request");
        requireGreedyRequest(request);
        T5EncoderOutput encoderOutput = encoderPipeline.encode(request.inputTokenIds());
        int[] generated = new int[request.maxNewTokens()];
        int generatedTokens = 0;
        int decoderTokenId = request.decoderStartTokenId();
        T5DecoderCache cache = T5DecoderCache.empty();
        for (int step = 0; step < request.maxNewTokens(); step++) {
            T5DecoderState state = decoderPipeline.decodeStep(decoderTokenId, encoderOutput, cache);
            float[] logits = lmHead.logits(state.lastHiddenState());
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
