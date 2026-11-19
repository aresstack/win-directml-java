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
    void offsetsTimestampSegmentsAfterTheFirstSixSecondWindow() throws Exception {
        Path model = Path.of("..", "model", "whisper-tiny");
        Path sample = model.resolve("whisper-de-speech.wav");
        Assumptions.assumeTrue(Files.isRegularFile(model.resolve("model.safetensors")));
        Assumptions.assumeTrue(Files.isRegularFile(sample));

        float[] speech = new WhisperWavAudioLoader().load(sample);
        float[] delayedSpeech = new float[6 * 16000 + speech.length];
        System.arraycopy(speech, 0, delayedSpeech, 6 * 16000, speech.length);
        TranscriptionResult result = new WhisperTranscriber(model)
                .transcribeShortRecordingWithTimestamps(delayedSpeech, 120);

        assertTrue(result.getText().toLowerCase(Locale.ROOT)
                .contains("dies ist ein test der deutschen spracherkennung"));
        assertTrue(result.getSegments().stream().anyMatch(segment ->
                        segment.getStartMillis() >= 6000
                                && segment.getEndMillis() <= delayedSpeech.length * 1000L / 16000),
                "The second audio window must retain the six-second offset");
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
