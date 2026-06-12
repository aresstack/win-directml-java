package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.generation.GenerationFinishReason;
import com.aresstack.windirectml.inference.generation.GenerationSummary;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Result boundary for the future T5 WARP generation runtime.
 */
public final class T5RuntimeResult {
    public enum FinishReason {
        unsupported,
        stop_token,
        max_tokens
    }

    /** Sentinel for {@link #finishTokenId()} when generation did not end on a stop token. */
    public static final int NO_FINISH_TOKEN = -1;

    private final int[] outputTokenIds;
    private final FinishReason finishReason;
    private final int generatedTokens;
    private final int finishTokenId;
    private final T5GenerationMetrics generationMetrics;

    public T5RuntimeResult(int[] outputTokenIds, FinishReason finishReason, int generatedTokens) {
        this(outputTokenIds, finishReason, generatedTokens, NO_FINISH_TOKEN, T5GenerationMetrics.empty());
    }

    public T5RuntimeResult(int[] outputTokenIds, FinishReason finishReason, int generatedTokens,
                           T5GenerationMetrics generationMetrics) {
        this(outputTokenIds, finishReason, generatedTokens, NO_FINISH_TOKEN, generationMetrics);
    }

    public T5RuntimeResult(int[] outputTokenIds, FinishReason finishReason, int generatedTokens,
                           int finishTokenId, T5GenerationMetrics generationMetrics) {
        this.outputTokenIds = outputTokenIds == null ? new int[0] : Arrays.copyOf(outputTokenIds, outputTokenIds.length);
        this.finishReason = Objects.requireNonNull(finishReason, "finishReason");
        if (generatedTokens < 0) {
            throw new IllegalArgumentException("generatedTokens must not be negative: " + generatedTokens);
        }
        this.generatedTokens = generatedTokens;
        this.finishTokenId = finishTokenId;
        this.generationMetrics = Objects.requireNonNull(generationMetrics, "generationMetrics");
    }

    public int[] outputTokenIds() {
        return Arrays.copyOf(outputTokenIds, outputTokenIds.length);
    }

    public FinishReason finishReason() {
        return finishReason;
    }

    public int generatedTokens() {
        return generatedTokens;
    }

    /**
     * The stop token that terminated generation (when {@link #finishReason()} is {@link FinishReason#stop_token}), or
     * {@link #NO_FINISH_TOKEN}. Harmonised with the decoder-only contract: this stop token is NOT part of
     * {@link #outputTokenIds()} and is not streamed.
     */
    public int finishTokenId() {
        return finishTokenId;
    }

    public T5GenerationMetrics generationMetrics() {
        return generationMetrics;
    }

    /**
     * Map this seq2seq result onto the model-family-neutral {@link GenerationSummary} so callers can read the same
     * fields as the decoder-only runtime. T5's finish-reason enum and its full {@link T5GenerationMetrics} are
     * preserved; the summary only selects the shared subset (encoder time is reported as the neutral prefill stage).
     *
     * @param promptTokenCount the input/prompt token count (T5RuntimeResult does not carry the input itself)
     */
    public GenerationSummary toSummary(int promptTokenCount) {
        List<Integer> ids = new ArrayList<>(outputTokenIds.length);
        for (int id : outputTokenIds) {
            ids.add(id);
        }
        return new GenerationSummary(
                ids,
                toGenerationFinishReason(finishReason),
                finishTokenId,
                promptTokenCount,
                generatedTokens,
                generationMetrics.runtimeNanos(),
                generationMetrics.encoderNanos(),
                generationMetrics.decodeNanos(),
                generationMetrics.lmHeadNanos());
    }

    private static GenerationFinishReason toGenerationFinishReason(FinishReason finishReason) {
        switch (finishReason) {
            case stop_token:
                return GenerationFinishReason.STOP_TOKEN;
            case max_tokens:
                return GenerationFinishReason.LENGTH;
            case unsupported:
            default:
                return GenerationFinishReason.UNSUPPORTED;
        }
    }
}
