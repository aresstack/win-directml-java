package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.generation.GenerationFinishReason;
import com.aresstack.windirectml.inference.generation.GenerationSummary;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * T5RuntimeResult maps onto the shared {@link GenerationSummary} without losing its T5-specific metrics or renaming
 * its finish-reason enum. Device-free.
 */
class T5RuntimeResultSummaryTest {

    private static T5GenerationMetrics metrics() {
        // tokenization, encoder, crossAttnPrepare, decode, lmHead, tokenSelect, detokenize, runtime, generatedTokens
        return new T5GenerationMetrics(10L, 400L, 25L, 300L, 100L, 7L, 5L, 1000L, 2);
    }

    @Test
    void stopTokenResultMapsToSharedSummary() {
        T5RuntimeResult result = new T5RuntimeResult(new int[]{5, 6}, T5RuntimeResult.FinishReason.stop_token,
                2, 99, metrics());

        GenerationSummary summary = result.toSummary(7);

        assertEquals(GenerationFinishReason.STOP_TOKEN, summary.finishReason());
        assertEquals(99, summary.finishTokenId());
        assertEquals(7, summary.promptTokenCount());
        assertEquals(2, summary.outputTokenCount());
        assertEquals(List.of(5, 6), summary.generatedTokenIds());
        assertEquals(1000L, summary.totalNanos());
        assertEquals(400L, summary.prefillNanos());   // T5 encoder time -> neutral prefill stage
        assertEquals(300L, summary.decodeNanos());
        assertEquals(100L, summary.lmHeadNanos());
    }

    @Test
    void maxTokensResultMapsToLengthWithNoFinishToken() {
        T5RuntimeResult result = new T5RuntimeResult(new int[]{1, 2}, T5RuntimeResult.FinishReason.max_tokens,
                2, metrics());

        GenerationSummary summary = result.toSummary(3);

        assertEquals(GenerationFinishReason.LENGTH, summary.finishReason());
        assertEquals(GenerationSummary.NO_FINISH_TOKEN, summary.finishTokenId());
        assertEquals(2, summary.outputTokenCount());
        assertEquals(List.of(1, 2), summary.generatedTokenIds());
    }

    @Test
    void t5SpecificMetricsArePreservedOnTheResult() {
        T5RuntimeResult result = new T5RuntimeResult(new int[]{1}, T5RuntimeResult.FinishReason.max_tokens,
                1, metrics());

        // The summary selects the shared subset; the full T5 metrics remain available on the result.
        assertEquals(25L, result.generationMetrics().crossAttentionPrepareNanos());
        assertEquals(7L, result.generationMetrics().tokenSelectionNanos());
        assertEquals(10L, result.generationMetrics().tokenizationNanos());
    }
}
