package com.aresstack.windirectml.inference.whisper;

import com.aresstack.windirectml.inference.whisper.audio.WhisperLogMelSpectrogram;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/** Exercise real pretrained Whisper Tiny weights without creating a fake transcript. */
class WhisperInferenceSmokeTest {
    @Test
    void runsMultilingualEncoderDecoderWithRealCheckpoint() throws Exception {
        Path modelDir = Path.of("..", "model", "whisper-tiny");
        Assumptions.assumeTrue(Files.isRegularFile(modelDir.resolve("model.safetensors")));
        WhisperModelConfig config = WhisperModelConfig.load(modelDir.resolve("config.json"));
        WhisperCheckpoint weights = WhisperCheckpoint.open(modelDir.resolve("model.safetensors"));
        WhisperTokenizer tokenizer = new WhisperTokenizer(modelDir);

        float[] samples = new float[16000];
        for (int i = 0; i < samples.length; i++)
            samples[i] = (float) (0.2 * Math.sin(2 * Math.PI * 440.0 * i / 16000));
        float[][] mel = new WhisperLogMelSpectrogram(config.melBands()).compute(samples);
        float[][] shortMel = new float[mel.length][40];
        for (int i = 0; i < mel.length; i++)
            System.arraycopy(mel[i], 0, shortMel[i], 0, shortMel[i].length);

        float[][] audio = new WhisperAudioEncoder(weights, config).encode(shortMel);
        assertEquals(20, audio.length);
        assertEquals(config.modelWidth(), audio[0].length);
        for (float value : audio[0]) assertTrue(Float.isFinite(value));

        WhisperTextDecoder.Session session = new WhisperTextDecoder(weights, config).newSession(audio);
        int[] prefix = {
                tokenizer.requireSpecial("<|startoftranscript|>"),
                tokenizer.requireSpecial("<|de|>"),
                tokenizer.requireSpecial("<|transcribe|>"),
                tokenizer.requireSpecial("<|notimestamps|>")
        };
        float[] scores = null;
        for (int token : prefix) scores = session.nextLogits(token);
        assertEquals(config.vocabularySize(), scores.length);
        for (float score : scores) assertTrue(Float.isFinite(score));
        assertEquals(4, session.tokensConsumed());
    }

    @Test
    void readsByteLevelWhisperVocabularyAndGermanSpecialTokens() throws Exception {
        Path modelDir = Path.of("..", "model", "whisper-tiny");
        Assumptions.assumeTrue(Files.isRegularFile(modelDir.resolve("vocab.json")));
        WhisperTokenizer tokenizer = new WhisperTokenizer(modelDir);
        assertEquals(50258, tokenizer.requireSpecial("<|startoftranscript|>"));
        assertEquals(50261, tokenizer.requireSpecial("<|de|>"));
        assertEquals(50359, tokenizer.requireSpecial("<|transcribe|>"));
        assertEquals(50363, tokenizer.requireSpecial("<|notimestamps|>"));
        assertEquals("!", tokenizer.decode(java.util.Arrays.asList(0)));
    }
}
