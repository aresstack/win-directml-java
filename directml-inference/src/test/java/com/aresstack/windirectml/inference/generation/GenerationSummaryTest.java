package com.aresstack.windirectml.inference.generation;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGenerationResult;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The shared generation summary/finish-reason contract. Device-free. Also demonstrates that a decoder-only result maps
 * cleanly onto the same neutral view (so the type is genuinely usable by both families), without modifying the
 * decoder-only block.
 */
class GenerationSummaryTest {

    @Test
    void mapsDecoderOnlyFinishReasons() {
        assertEquals(GenerationFinishReason.STOP_TOKEN, GenerationFinishReason.fromDecoderOnlyReason("eos_token"));
        assertEquals(GenerationFinishReason.LENGTH, GenerationFinishReason.fromDecoderOnlyReason("length"));
        assertEquals(GenerationFinishReason.UNSUPPORTED, GenerationFinishReason.fromDecoderOnlyReason("nope"));
    }

    @Test
    void validatesCountsAndExposesMillis() {
        GenerationSummary summary = new GenerationSummary(List.of(4, 5), GenerationFinishReason.STOP_TOKEN, 9,
                3, 2, 2_000_000L, 400_000L, 300_000L, 100_000L);
        assertEquals(2L, summary.totalMillis());
        assertEquals(0L, summary.prefillMillis());
        assertEquals(List.of(4, 5), summary.generatedTokenIds());

        assertThrows(IllegalArgumentException.class, () -> new GenerationSummary(List.of(),
                GenerationFinishReason.LENGTH, -1, -1, 0, 0L, 0L, 0L, 0L));
        assertThrows(IllegalArgumentException.class, () -> new GenerationSummary(List.of(),
                GenerationFinishReason.LENGTH, -1, 0, -1, 0L, 0L, 0L, 0L));
    }

    @Test
    void decoderOnlyResultMapsOntoTheSharedSummary() {
        // Uses the additive DecoderOnlyGenerationResult.toSummary() so both families produce GenerationSummary.
        DecoderOnlyGenerationResult result = new DecoderOnlyGenerationResult(
                List.of(1, 2, 3), List.of(4, 5), List.of(1, 2, 3, 4, 5), 2, "eos_token", 9, 16,
                1000L, 400L, 300L, 100L, 5L, List.of(), List.of());

        GenerationSummary summary = result.toSummary();

        assertEquals(GenerationFinishReason.STOP_TOKEN, summary.finishReason());
        assertEquals(9, summary.finishTokenId());
        assertEquals(3, summary.promptTokenCount());
        assertEquals(2, summary.outputTokenCount());
        assertEquals(List.of(4, 5), summary.generatedTokenIds());
        assertEquals(1000L, summary.totalNanos());
        assertEquals(400L, summary.prefillNanos());
        assertEquals(300L, summary.decodeNanos());
        assertEquals(100L, summary.lmHeadNanos());
    }
}
