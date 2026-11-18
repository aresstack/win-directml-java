package com.aresstack.windirectml.inference.whisper;

import com.aresstack.windirectml.inference.whisper.audio.WhisperLogMelSpectrogram;
import com.aresstack.windirectml.inference.whisper.audio.WhisperWavAudioLoader;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Execute multilingual Whisper speech-to-text completely inside the Java 21 JVM.
 * Keep the transcription use case independent of the file decoder and backend.
 *
 * The first implementation uses greedy decoding and fixed 30-second segments;
 * timestamp-aligned boundary recovery remains a separate enhancement.
 */
public final class WhisperTranscriber {
    private static final int CHUNK_LENGTH = WhisperLogMelSpectrogram.SAMPLES_PER_CHUNK;

    private final WhisperModelConfig config;
    private final WhisperAudioEncoder encoder;
    private final WhisperTextDecoder decoder;
    private final WhisperTokenizer tokenizer;
    private final WhisperLogMelSpectrogram frontend;

    public WhisperTranscriber(Path modelDirectory) throws IOException {
        Objects.requireNonNull(modelDirectory, "modelDirectory");
        config = WhisperModelConfig.load(modelDirectory.resolve("config.json"));
        WhisperCheckpoint checkpoint = WhisperCheckpoint.open(modelDirectory.resolve("model.safetensors"));
        encoder = new WhisperAudioEncoder(checkpoint, config);
        decoder = new WhisperTextDecoder(checkpoint, config);
        tokenizer = new WhisperTokenizer(modelDirectory);
        frontend = new WhisperLogMelSpectrogram(config.melBands());
    }

    /** Decode a WAV recording into 16 kHz mono float PCM, then transcribe German speech. */
    public String transcribe(Path wavFile, int maximumTokensPerChunk) throws IOException {
        return transcribe(new WhisperWavAudioLoader().load(wavFile), maximumTokensPerChunk);
    }

    /** Accept already decoded 16-kHz mono float PCM from another audio pipeline. */
    public String transcribe(float[] audioSamples, int maximumTokensPerChunk) throws IOException {
        Objects.requireNonNull(audioSamples, "audioSamples");
        if (maximumTokensPerChunk < 1 || maximumTokensPerChunk > config.textContext() - 4) {
            throw new IllegalArgumentException("Invalid Whisper token limit");
        }
        StringBuilder transcript = new StringBuilder();
        for (int start = 0; start < audioSamples.length; start += CHUNK_LENGTH) {
            float[][] mel = frontend.compute(audioSamples, start);
            String text = transcribeMel(mel, maximumTokensPerChunk);
            if (!text.isBlank()) {
                if (transcript.length() > 0) transcript.append(' ');
                transcript.append(text.strip());
            }
        }
        return transcript.toString();
    }

    /**
     * Transcribe a prepared mel spectrogram, using smaller frame counts for
     * focused integration tests and the full 3000 frames for normal audio.
     */
    public String transcribeMel(float[][] mel, int maximumTokens) throws IOException {
        float[][] audioFeatures = encoder.encode(mel);
        WhisperTextDecoder.Session session = decoder.newSession(audioFeatures);
        int[] prefix = {
                tokenizer.requireSpecial("<|startoftranscript|>"),
                tokenizer.requireSpecial("<|de|>"),
                tokenizer.requireSpecial("<|transcribe|>"),
                tokenizer.requireSpecial("<|notimestamps|>")
        };
        float[] logits = null;
        for (int token : prefix) logits = session.nextLogits(token);
        int endOfText = tokenizer.requireSpecial("<|endoftext|>");
        List<Integer> tokens = new ArrayList<Integer>();
        for (int i = 0; i < maximumTokens; i++) {
            int next = selectTextToken(logits, endOfText);
            if (next == endOfText) break;
            tokens.add(next);
            if (session.tokensConsumed() >= config.textContext()) break;
            logits = session.nextLogits(next);
        }
        return tokenizer.decode(tokens).strip();
    }

    private int selectTextToken(float[] logits, int endOfText) {
        int best = endOfText;
        float bestScore = logits[endOfText];
        for (int token = 0; token < tokenizer.textTokenCount(); token++) {
            if (logits[token] > bestScore) {
                bestScore = logits[token];
                best = token;
            }
        }
        return best;
    }
}
