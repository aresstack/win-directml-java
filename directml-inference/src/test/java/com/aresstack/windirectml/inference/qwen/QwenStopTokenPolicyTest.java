package com.aresstack.windirectml.inference.qwen;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link QwenStopTokenPolicy}.
 *
 * <p>Verifies that the stop policy correctly identifies Qwen EOS tokens
 * and rejects non-stop tokens.
 */
class QwenStopTokenPolicyTest {

    @Test
    void endOfTextStopsGeneration() {
        assertTrue(QwenStopTokenPolicy.shouldStop(QwenStopTokenPolicy.ENDOFTEXT_ID));
    }

    @Test
    void imEndStopsGeneration() {
        assertTrue(QwenStopTokenPolicy.shouldStop(QwenStopTokenPolicy.IM_END_ID));
    }

    @Test
    void regularTokenDoesNotStop() {
        assertFalse(QwenStopTokenPolicy.shouldStop(0));
        assertFalse(QwenStopTokenPolicy.shouldStop(100));
        assertFalse(QwenStopTokenPolicy.shouldStop(1000));
        assertFalse(QwenStopTokenPolicy.shouldStop(151644)); // im_start is NOT a stop token
    }

    @Test
    void stopTokenIdsListContainsExpectedTokens() {
        List<Integer> ids = QwenStopTokenPolicy.stopTokenIds();
        assertEquals(2, ids.size());
        assertTrue(ids.contains(151643)); // endoftext
        assertTrue(ids.contains(151645)); // im_end
    }

    @Test
    void stopTokenIdsAreConsistentWithShouldStop() {
        for (int id : QwenStopTokenPolicy.stopTokenIds()) {
            assertTrue(QwenStopTokenPolicy.shouldStop(id),
                    "shouldStop must return true for id " + id);
        }
    }

    @Test
    void stopTokenIdsIsUnmodifiable() {
        assertThrows(UnsupportedOperationException.class,
                () -> QwenStopTokenPolicy.stopTokenIds().add(999));
    }

    @Test
    void policyMatchesTokenizerIsEos() {
        // QwenStopTokenPolicy and QwenTokenizer.isEos should agree
        assertTrue(QwenStopTokenPolicy.shouldStop(QwenTokenizer.ENDOFTEXT_ID));
        assertTrue(QwenStopTokenPolicy.shouldStop(QwenTokenizer.IM_END_ID));
        assertFalse(QwenStopTokenPolicy.shouldStop(QwenTokenizer.IM_START_ID));
    }
}
