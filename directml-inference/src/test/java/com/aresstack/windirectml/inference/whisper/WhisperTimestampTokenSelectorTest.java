package com.aresstack.windirectml.inference.whisper;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;

class WhisperTimestampTokenSelectorTest {
    private final WhisperTimestampTokenSelector selector =
            new WhisperTimestampTokenSelector(100, 200, 5, 5);

    @Test
    void letsWhisperFinishAfterRecognizedText() {
        float[] logits = scores();
        logits[5] = 10;
        logits[0] = 8;
        logits[120] = 0;

        assertEquals(5, selector.select(logits, Arrays.asList(100, 0), 200));
    }

    @Test
    void forcesClosingTimestampAfterTextAndTimestamp() {
        float[] logits = scores();
        logits[1] = 20;
        logits[120] = 5;
        logits[5] = 0;

        assertEquals(120, selector.select(logits, Arrays.asList(100, 0, 119), 200));
    }

    @Test
    void requiresTextAfterTwoConsecutiveTimestamps() {
        float[] logits = scores();
        logits[150] = 20;
        logits[1] = 5;

        assertEquals(1, selector.select(logits, Arrays.asList(100, 0, 110, 110), 200));
    }

    @Test
    void startsWithTimestampWithinFirstSecond() {
        float[] logits = scores();
        logits[180] = 20;
        logits[120] = 5;
        logits[1] = 10;

        assertEquals(120, selector.select(logits, Collections.emptyList(), 200));
    }

    @Test
    void prefersTotalTimestampProbabilityOverSingleTextToken() {
        float[] logits = scores();
        logits[0] = 1.5f;
        for (int token = 101; token <= 110; token++) {
            logits[token] = 0;
        }

        assertEquals(101, selector.select(logits, Arrays.asList(100, 0), 200));
    }

    @Test
    void neverLetsSegmentTimestampMoveBackward() {
        float[] logits = scores();
        logits[101] = 20;
        logits[141] = 3;
        logits[1] = -1;

        assertEquals(141, selector.select(logits, Arrays.asList(100, 0, 140, 140, 1), 200));
    }

    @Test
    void masksSuppressedTokensAndBlankAtTheBeginningWithoutTimestamps() {
        WhisperTimestampTokenSelector rules = new WhisperTimestampTokenSelector(
                100, 200, 5, 5, new int[]{2}, new int[]{3, 5});
        float[] logits = scores();
        logits[2] = 30;
        logits[3] = 20;
        logits[5] = 15;
        logits[150] = 40;
        logits[1] = 1;

        float[] first = rules.constrain(logits, Collections.emptyList(), 200, false);
        assertEquals(1, WhisperTimestampTokenSelector.argmax(first));
        assertEquals(Float.NEGATIVE_INFINITY, first[150], "Timestamps are disabled");

        float[] later = rules.constrain(logits, Arrays.asList(1), 200, false);
        assertEquals(3, WhisperTimestampTokenSelector.argmax(later));
        assertEquals(Float.NEGATIVE_INFINITY, later[2]);
    }

    @Test
    void forbidsTimestampsAfterTheEndOfTheRecordedAudio() {
        float[] logits = scores();
        logits[180] = 20;
        logits[5] = 1;

        float[] masked = selector.constrain(logits, Arrays.asList(100, 0), 150, true);
        assertEquals(Float.NEGATIVE_INFINITY, masked[180]);
        assertEquals(5, WhisperTimestampTokenSelector.argmax(masked));
    }

    private static float[] scores() {
        float[] scores = new float[201];
        Arrays.fill(scores, -1000);
        return scores;
    }
}
