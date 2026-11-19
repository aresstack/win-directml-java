package com.aresstack.windirectml.inference.whisper;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WhisperRepetitionGuardTest {
    private final WhisperRepetitionGuard guard = new WhisperRepetitionGuard(
            100, 200, 300, WhisperRepetitionGuardTest::decodeTokens);

    @Test
    void removesRepeatedSingleWordAfterItsFirstOccurrence() {
        List<Integer> emitted = new ArrayList<>(Arrays.asList(200));
        for (int repetition = 0; repetition < 5; repetition++) {
            emitted.addAll(Arrays.asList(10, 11, 12));
        }

        assertTrue(guard.stopAndTrimRepeatedText(emitted));
        assertEquals(Arrays.asList(200, 10, 11, 12), emitted);
    }

    @Test
    void keepsOnlyTheFirstWordWhenItsInitialTokenizationDiffers() {
        List<Integer> emitted = new ArrayList<>(Arrays.asList(200, 20, 21));
        for (int repetition = 0; repetition < 5; repetition++) {
            emitted.addAll(Arrays.asList(10, 11));
        }

        assertTrue(guard.stopAndTrimRepeatedText(emitted));
        assertEquals(Arrays.asList(200, 20, 21), emitted);
    }

    @Test
    void preservesEarlierTimestampedSegmentsWhenTheModelLoops() {
        List<Integer> emitted = new ArrayList<>(Arrays.asList(
                200, 20, 21, 220, 220, 30, 31, 230, 230));
        for (int repetition = 0; repetition < 5; repetition++) {
            emitted.addAll(Arrays.asList(10, 11, 12));
        }

        assertTrue(guard.stopAndTrimRepeatedText(emitted));
        assertEquals(Arrays.asList(200, 20, 21, 220, 220, 30, 31, 230, 230, 10, 11, 12),
                emitted);
    }

    @Test
    void doesNotStopOrdinaryRepeatedSpeechPrematurely() {
        List<Integer> emitted = new ArrayList<>(Arrays.asList(200));
        for (int repetition = 0; repetition < 4; repetition++) {
            emitted.addAll(Arrays.asList(10, 11));
        }

        assertFalse(guard.stopAndTrimRepeatedText(emitted));
    }

    @Test
    void doesNotCrossTimestampBoundariesToDetectRepetitions() {
        List<Integer> emitted = new ArrayList<>(Arrays.asList(
                200, 10, 11, 220, 220, 10, 11, 240, 240, 10, 11));

        assertFalse(guard.stopAndTrimRepeatedText(emitted));
    }

    @Test
    void doesNotTreatDifferentSentencesAsOneRepeatedPhrase() {
        List<Integer> emitted = new ArrayList<>(Arrays.asList(
                200, 10, 11, 10, 11, 10, 11, 10, 11, 10, 12));
        assertFalse(guard.stopAndTrimRepeatedText(emitted));
    }

    private static String decodeTokens(List<Integer> tokens) {
        StringBuilder text = new StringBuilder();
        for (Integer token : tokens) {
            if (token == 10) {
                text.append(" Kakuhr");
            } else if (token == 20) {
                text.append("Kakuhr");
            } else if (token == 11 || token == 21) {
                text.append('.');
            } else {
                text.append('[').append(token).append(']');
            }
        }
        return text.toString();
    }
}
