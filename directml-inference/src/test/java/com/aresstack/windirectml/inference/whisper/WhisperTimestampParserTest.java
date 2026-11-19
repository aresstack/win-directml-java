package com.aresstack.windirectml.inference.whisper;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class WhisperTimestampParserTest {
    private final WhisperTimestampParser parser =
            new WhisperTimestampParser(100, 1600, 10, this::decodeText);

    @Test
    void preservesModelEmittedIntervalsAndAppliesWindowOffset() {
        TranscriptionResult result = parser.parse(
                Arrays.asList(100, 0, 1, 140, 140, 2, 3, 190), 60000, 6000);

        assertEquals("Hello worldsecond sentence", result.getText());
        assertEquals(2, result.getSegments().size());
        assertEquals(60000, result.getSegments().get(0).getStartMillis());
        assertEquals(60800, result.getSegments().get(0).getEndMillis());
        assertEquals("Hello world", result.getSegments().get(0).getText());
        assertEquals(60800, result.getSegments().get(1).getStartMillis());
        assertEquals(61800, result.getSegments().get(1).getEndMillis());
        assertEquals("second sentence", result.getSegments().get(1).getText());
        assertFalse(result.hasUnalignedText());
    }

    @Test
    void doesNotInventEndTimeForIncompleteSegment() {
        TranscriptionResult result = parser.parse(Arrays.asList(100, 0, 1), 5000, 3000);
        assertTrue(result.getSegments().isEmpty());
        assertEquals("Hello world", result.getUnalignedText());
    }

    @Test
    void clampsTimestampToAvailableRecordingDuration() {
        TranscriptionResult result = parser.parse(Arrays.asList(100, 0, 400), 30000, 500);
        assertEquals(1, result.getSegments().size());
        assertEquals(30500, result.getSegments().get(0).getEndMillis());
    }

    @Test
    void preservesUnalignedTextWithoutCreatingSyntheticTimestamps() {
        TranscriptionResult result = parser.parse(Arrays.asList(0, 100, 1, 120), 0, 6000);
        assertEquals("Hello", result.getUnalignedText());
        assertEquals("world", result.getSegments().get(0).getText());
    }

    private String decodeText(List<Integer> ids) {
        StringBuilder text = new StringBuilder();
        for (int id : ids) {
            if (id == 0) text.append("Hello ");
            else if (id == 1) text.append("world");
            else if (id == 2) text.append("second ");
            else if (id == 3) text.append("sentence");
        }
        return text.toString();
    }
}
