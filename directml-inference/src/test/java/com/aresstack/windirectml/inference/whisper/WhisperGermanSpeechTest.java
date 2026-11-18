package com.aresstack.windirectml.inference.whisper;

import com.aresstack.windirectml.inference.whisper.audio.WhisperLogMelSpectrogram;
import com.aresstack.windirectml.inference.whisper.audio.WhisperWavAudioLoader;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verify that real spoken German produces text with pretrained multilingual Whisper.
 * Keep the locally synthesized WAV and model weights outside version control.
 */
class WhisperGermanSpeechTest {
    @Test
    void transcribesLocallySynthesizedGermanSpeech() throws Exception {
        Path modelDir = Path.of("..", "model", "whisper-tiny");
        Path wav = modelDir.resolve("whisper-de-speech.wav");
        Assumptions.assumeTrue(Files.isRegularFile(wav));
        Assumptions.assumeTrue(Files.isRegularFile(modelDir.resolve("model.safetensors")));

        float[] samples = new WhisperWavAudioLoader().load(wav);
        float[][] fullMel = new WhisperLogMelSpectrogram(80).compute(samples);
        int frames = Math.min(3000, (samples.length + 159) / 160 + 60);
        float[][] mel = new float[80][frames];
        for (int band = 0; band < mel.length; band++) {
            System.arraycopy(fullMel[band], 0, mel[band], 0, frames);
        }

        String transcript = new WhisperTranscriber(modelDir).transcribeMel(mel, 36);
        System.out.println("WHISPER_GERMAN_EXPECTED: Hallo, dies ist ein Test der deutschen Spracherkennung.");
        System.out.println("WHISPER_GERMAN_ACTUAL: " + transcript);
        assertTrue(transcript.toLowerCase(java.util.Locale.ROOT)
                .contains("dies ist ein test der deutschen spracherkennung"),
                "Whisper transcription differs from expected German speech: " + transcript);
    }
}
