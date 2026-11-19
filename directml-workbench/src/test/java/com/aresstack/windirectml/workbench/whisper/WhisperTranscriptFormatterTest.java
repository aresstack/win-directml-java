package com.aresstack.windirectml.workbench.whisper;

import com.aresstack.windirectml.inference.whisper.TranscriptionResult;
import com.aresstack.windirectml.inference.whisper.TranscriptionSegment;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;

class WhisperTranscriptFormatterTest {
    @Test
    void showsMinuteSecondAndCentisecondIntervals() {
        TranscriptionResult result = new TranscriptionResult("Hallo!", "",
                Arrays.asList(new TranscriptionSegment(62000, 63920, "Hallo!")));
        assertEquals("[01:02.00 – 01:03.92] Hallo!", WhisperTranscriptFormatter.format(result));
    }

    @Test
    void warnsWhenTimestampPairsAreMissing() {
        TranscriptionResult result = new TranscriptionResult("Hallo!", "Hallo!",
                Collections.emptyList());
        assertTrue(WhisperTranscriptFormatter.format(result).contains("Ohne bestätigte Zeitzuordnung"));
    }
}
