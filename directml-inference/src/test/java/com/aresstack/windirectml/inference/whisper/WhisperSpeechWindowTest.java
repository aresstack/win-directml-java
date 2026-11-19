package com.aresstack.windirectml.inference.whisper;

import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class WhisperSpeechWindowTest {
    @Test
    void skipsPureSilenceWithoutTranscribingIt() {
        assertFalse(WhisperSpeechWindow.detect(new float[16000], 0, 16000).isPresent());
    }

    @Test
    void preservesAbsoluteOffsetAfterLongLeadingSilence() {
        float[] audio = new float[12 * 16000];
        for (int i = 6 * 16000; i < 10 * 16000; i++) {
            audio[i] = (float) (0.25 * Math.sin(2 * Math.PI * 440 * i / 16000.0));
        }

        Optional<WhisperSpeechWindow> selected =
                WhisperSpeechWindow.detect(audio, 0, audio.length);
        assertTrue(selected.isPresent());
        assertEquals(5 * 16000 + 14400, selected.get().firstSample());
        assertEquals(4 * 16000 + 3200, selected.get().sampleCount());
    }

    @Test
    void doesNotShiftWindowOutsideRequestedAudioRange() {
        float[] audio = new float[16000];
        for (int i = 0; i < 16000; i++) {
            audio[i] = 0.2f;
        }
        WhisperSpeechWindow window = WhisperSpeechWindow.detect(audio, 0, audio.length).orElseThrow();
        assertEquals(0, window.firstSample());
        assertEquals(audio.length, window.sampleCount());
    }

    @Test
    void splitsRepeatedSpeechAtSilentGapsWithoutResettingTheTimeline() {
        float[] audio = new float[24 * 16000];
        for (int occurrence = 0; occurrence < 4; occurrence++) {
            int start = occurrence * 6 * 16000;
            for (int i = start; i < start + 4 * 16000; i++) {
                audio[i] = (float) (0.2 * Math.sin(2 * Math.PI * 440 * i / 16000.0));
            }
        }

        java.util.List<WhisperSpeechWindow> windows = WhisperSpeechWindow.split(audio, 30 * 16000);
        assertEquals(4, windows.size());
        assertEquals(0, windows.get(0).firstSample());
        assertTrue(windows.get(1).firstSample() >= 5 * 16000);
        assertTrue(windows.get(2).firstSample() >= 11 * 16000);
        assertTrue(windows.get(3).firstSample() >= 17 * 16000);
        assertTrue(windows.stream().allMatch(window ->
                window.sampleCount() <= 30 * 16000));
    }

    @Test
    void splitsUninterruptedSpeechBeforeTheWhisperThirtySecondLimit() {
        float[] audio = new float[65 * 16000];
        java.util.Arrays.fill(audio, 0.2f);

        java.util.List<WhisperSpeechWindow> windows = WhisperSpeechWindow.split(audio, 30 * 16000);

        assertEquals(3, windows.size());
        assertEquals(30 * 16000, windows.get(1).firstSample());
        assertEquals(65 * 16000, windows.get(2).firstSample() + windows.get(2).sampleCount());
    }

    @Test
    void rejectsInvalidIntervals() {
        assertThrows(IllegalArgumentException.class,
                () -> WhisperSpeechWindow.detect(new float[100], 99, 5));
    }
}
