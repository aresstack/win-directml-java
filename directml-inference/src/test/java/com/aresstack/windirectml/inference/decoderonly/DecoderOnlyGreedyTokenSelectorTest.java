package com.aresstack.windirectml.inference.decoderonly;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DecoderOnlyGreedyTokenSelectorTest {

    @Test
    void selectsHighestLogit() {
        DecoderOnlyGreedyTokenSelector selector = new DecoderOnlyGreedyTokenSelector(tokenId -> false);
        DecoderOnlyGeneratedTokens generatedTokens = new DecoderOnlyGeneratedTokens(4);
        float[] logits = {0.1f, 2.0f, 1.5f};

        assertEquals(1, selector.selectNextToken(logits, generatedTokens, 1.0f));
    }

    @Test
    void appliesRepetitionPenaltyBeforeSelection() {
        DecoderOnlyGreedyTokenSelector selector = new DecoderOnlyGreedyTokenSelector(tokenId -> false);
        DecoderOnlyGeneratedTokens generatedTokens = new DecoderOnlyGeneratedTokens(4);
        generatedTokens.add(1);
        float[] logits = {0.1f, 2.0f, 1.5f};

        assertEquals(2, selector.selectNextToken(logits, generatedTokens, 2.0f));
    }

    @Test
    void delegatesStopTokenDecision() {
        DecoderOnlyGreedyTokenSelector selector = new DecoderOnlyGreedyTokenSelector(tokenId -> tokenId == 99);

        assertTrue(selector.shouldStop(99));
        assertFalse(selector.shouldStop(98));
    }
}
