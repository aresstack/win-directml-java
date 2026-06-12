package com.aresstack.windirectml.inference.generation;

import java.util.List;
import java.util.Objects;

/**
 * Model-family-neutral view of one generation run.
 *
 * <p>The common subset both the decoder-only and the seq2seq (T5) runtimes can deliver, so workbench/API can read the
 * same fields without family-specific logic: produced token ids, finish reason + finish token id, prompt/output token
 * counts and coarse stage timings. Families keep their own richer result/metrics types and map onto this view
 * (T5 keeps encoder/cross-attention/token-select metrics; decoder-only keeps its micro-profile).</p>
 *
 * <p>{@code prefillNanos} is the one-shot input-processing stage: the decoder-only prefill, or the seq2seq encoder.
 * {@code finishTokenId} is the terminating stop token (or {@link #NO_FINISH_TOKEN} when generation ended by length).</p>
 */
public record GenerationSummary(
        List<Integer> generatedTokenIds,
        GenerationFinishReason finishReason,
        int finishTokenId,
        int promptTokenCount,
        int outputTokenCount,
        long totalNanos,
        long prefillNanos,
        long decodeNanos,
        long lmHeadNanos) {

    /** Sentinel for {@link #finishTokenId()} when generation did not end on a stop token. */
    public static final int NO_FINISH_TOKEN = -1;

    public GenerationSummary {
        generatedTokenIds = generatedTokenIds == null ? List.of() : List.copyOf(generatedTokenIds);
        Objects.requireNonNull(finishReason, "finishReason");
        if (promptTokenCount < 0) {
            throw new IllegalArgumentException("promptTokenCount must not be negative: " + promptTokenCount);
        }
        if (outputTokenCount < 0) {
            throw new IllegalArgumentException("outputTokenCount must not be negative: " + outputTokenCount);
        }
    }

    public long totalMillis() {
        return totalNanos / 1_000_000L;
    }

    /** Prefill (decoder-only) or encoder (seq2seq) time in milliseconds. */
    public long prefillMillis() {
        return prefillNanos / 1_000_000L;
    }

    public long decodeMillis() {
        return decodeNanos / 1_000_000L;
    }

    public long lmHeadMillis() {
        return lmHeadNanos / 1_000_000L;
    }
}
