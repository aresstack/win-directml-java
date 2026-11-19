package com.aresstack.windirectml.inference.whisper;

import com.aresstack.windirectml.inference.whisper.audio.WhisperWavAudioLoader;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.*;

class WhisperTimestampSpeechTest {
    @Test
    void retainsAbsoluteTimelineForSpeechAfterSixSecondsOfSilence() throws Exception {
        Path model = Path.of("..", "model", "whisper-tiny");
        Path sample = model.resolve("whisper-de-speech.wav");
        Assumptions.assumeTrue(Files.isRegularFile(model.resolve("model.safetensors")));
        Assumptions.assumeTrue(Files.isRegularFile(sample));

        float[] speech = new WhisperWavAudioLoader().load(sample);
        float[] delayedSpeech = new float[6 * 16000 + speech.length];
        System.arraycopy(speech, 0, delayedSpeech, 6 * 16000, speech.length);
        TranscriptionResult result = new WhisperTranscriber(model)
                .transcribeShortRecordingWithTimestamps(delayedSpeech, 120);
        System.out.println("TIMESTAMP_DELAYED_TEXT: " + result.getText());
        System.out.println("TIMESTAMP_DELAYED_UNALIGNED: " + result.getUnalignedText());
        for (TranscriptionSegment segment : result.getSegments()) {
            System.out.println("TIMESTAMP_DELAYED_SEGMENT: "
                    + segment.getStartMillis() + " - " + segment.getEndMillis()
                    + ": " + segment.getText());
        }

        assertTrue(result.getText().toLowerCase(Locale.ROOT)
                .contains("dies ist ein test der deutschen spracherkennung"));
        assertTrue(result.getSegments().stream().anyMatch(segment ->
                        segment.getStartMillis() >= 5800
                                && segment.getStartMillis() <= 7000
                                && segment.getEndMillis() <= delayedSpeech.length * 1000L / 16000),
                "Speech detected after initial silence must retain its absolute recording position");
    }

    @Test
    void transcribesRepeatedSpeechAcrossMultipleIntervalsWithoutRunawayRepetition() throws Exception {
        Path model = Path.of("..", "model", "whisper-tiny");
        Path sample = model.resolve("whisper-de-speech.wav");
        Assumptions.assumeTrue(Files.isRegularFile(model.resolve("model.safetensors")));
        Assumptions.assumeTrue(Files.isRegularFile(sample));

        float[] speech = new WhisperWavAudioLoader().load(sample);
        float[] recording = new float[speech.length * 4];
        for (int i = 0; i < 4; i++) {
            System.arraycopy(speech, 0, recording, i * speech.length, speech.length);
        }

        TranscriptionResult result = new WhisperTranscriber(model)
                .transcribeShortRecordingWithTimestamps(recording, 400);

        System.out.println("TIMESTAMP_LONG_TEXT: " + result.getText());
        System.out.println("TIMESTAMP_LONG_UNALIGNED: " + result.getUnalignedText());
        for (TranscriptionSegment segment : result.getSegments()) {
            System.out.println("TIMESTAMP_LONG_SEGMENT: " + segment.getStartMillis()
                    + " - " + segment.getEndMillis() + ": " + segment.getText());
        }

        assertTrue(result.getText().toLowerCase(Locale.ROOT).contains("deutschen spracherkennung"));
        assertFalse(result.getSegments().isEmpty(), "Expected at least one aligned segment");
        assertTrue(result.getText().length() < 900, "Decoder appears stuck in a repetition loop");
        assertTrue(result.getSegments().stream().allMatch(segment ->
                        segment.getEndMillis() <= recording.length * 1000L / 16000),
                "Segment exceeds the recording duration");
    }

    @Test
    void recognizesGermanSpeechAndReturnsModelConfirmedTimestampSegments() throws Exception {
        Path model = Path.of("..", "model", "whisper-tiny");
        Path sample = model.resolve("whisper-de-speech.wav");
        Assumptions.assumeTrue(Files.isRegularFile(model.resolve("model.safetensors")));
        Assumptions.assumeTrue(Files.isRegularFile(sample));

        float[] audio = new WhisperWavAudioLoader().load(sample);
        TranscriptionResult result = new WhisperTranscriber(model)
                .transcribeShortRecordingWithTimestamps(audio, 120);

        System.out.println("TIMESTAMP_SPEECH: " + result.getText());
        for (TranscriptionSegment segment : result.getSegments()) {
            System.out.println("TIMESTAMP_SEGMENT: " + segment.getStartMillis() + " - "
                    + segment.getEndMillis() + ": " + segment.getText());
        }
        System.out.println("TIMESTAMP_UNALIGNED: " + result.getUnalignedText());
        assertTrue(result.getText().toLowerCase(Locale.ROOT)
                .contains("dies ist ein test der deutschen spracherkennung"));
        assertFalse(result.getSegments().isEmpty(), "The model did not emit timestamp pairs");
        assertTrue(result.getSegments().get(0).getStartMillis() >= 0);
        assertTrue(result.getSegments().get(0).getEndMillis() <= audio.length * 1000L / 16000);
    }
}
