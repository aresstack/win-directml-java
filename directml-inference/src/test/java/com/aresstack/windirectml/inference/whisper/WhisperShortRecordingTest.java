package com.aresstack.windirectml.inference.whisper;

import com.aresstack.windirectml.inference.whisper.audio.WhisperWavAudioLoader;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.assertTrue;

/** Confirm that short spoken German takes the same path as microphone audio. */
class WhisperShortRecordingTest {
    @Test
    void recognizesRealGermanSpeechThroughTheDeprecatedShortRecordingApi() throws Exception {
        Path model = Path.of("..", "model", "whisper-tiny");
        Path sample = model.resolve("whisper-de-speech.wav");
        Assumptions.assumeTrue(Files.isRegularFile(model.resolve("model.safetensors")));
        Assumptions.assumeTrue(Files.isRegularFile(sample));

        float[] audio = new WhisperWavAudioLoader().load(sample);
        String result = new WhisperTranscriber(model).transcribeShortRecording(audio, 80);

        System.out.println("WHISPER_SHORT_RECORDING: " + result);
        assertTrue(result.toLowerCase(Locale.ROOT)
                        .contains("dies ist ein test der deutschen spracherkennung"),
                "Unexpected transcription: " + result);
    }
}
