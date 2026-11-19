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
 * Preserve the original plain-text methods and expose timestamp-token decoding
 * through independent, additive segment-returning methods.
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
     * Transcribe a short, already decoded 16-kHz mono microphone recording.
     * Process six-second chunks to keep the pure Java reference backend responsive.
     * Include a small silence tail for the last spoken syllable.
     */
    public String transcribeShortRecording(float[] samples, int maximumTokensPerChunk) throws IOException {
        Objects.requireNonNull(samples, "samples");
        if (maximumTokensPerChunk < 1 || maximumTokensPerChunk > config.textContext() - 4) {
            throw new IllegalArgumentException("Invalid Whisper token limit");
        }
        int samplesPerPart = WhisperLogMelSpectrogram.SAMPLE_RATE * 6;
        StringBuilder result = new StringBuilder();
        for (int offset = 0; offset < samples.length; offset += samplesPerPart) {
            int sampleCount = Math.min(samplesPerPart, samples.length - offset);
            float[] chunk = java.util.Arrays.copyOfRange(samples, offset, offset + sampleCount);
            float[][] fullMel = frontend.compute(chunk);
            int frames = Math.min(WhisperLogMelSpectrogram.FRAMES_PER_CHUNK,
                    Math.max(40, (sampleCount + WhisperLogMelSpectrogram.HOP_LENGTH - 1)
                            / WhisperLogMelSpectrogram.HOP_LENGTH + 60));
            float[][] shortMel = new float[config.melBands()][frames];
            for (int band = 0; band < config.melBands(); band++) {
                System.arraycopy(fullMel[band], 0, shortMel[band], 0, frames);
            }
            String text = transcribeMel(shortMel, maximumTokensPerChunk);
            if (!text.isBlank()) {
                if (result.length() > 0) {
                    result.append(' ');
                }
                result.append(text.strip());
            }
        }
        return result.toString();
    }

    /** Return Whisper-generated speech intervals for a WAV recording. */
    public TranscriptionResult transcribeWithTimestamps(Path wavFile, int maximumTokensPerChunk)
            throws IOException {
        return transcribeWithTimestamps(new WhisperWavAudioLoader().load(wavFile), maximumTokensPerChunk);
    }

    /** Transcribe 16-kHz mono PCM using timestamp-driven 30-second windows. */
    public TranscriptionResult transcribeWithTimestamps(float[] samples, int maximumTokensPerChunk)
            throws IOException {
        Objects.requireNonNull(samples, "samples");
        validateTokenLimit(maximumTokensPerChunk);
        WhisperTimestampWindowStitcher stitcher = new WhisperTimestampWindowStitcher(samples.length);
        WhisperTimestampDecoder timestampDecoder = newTimestampDecoder();
        while (stitcher.hasNextWindow()) {
            int offset = stitcher.nextStartSample();
            int count = Math.min(CHUNK_LENGTH, samples.length - offset);
            float[][] mel = frontend.compute(samples, offset);
            stitcher.accept(timestampDecoder.transcribe(mel, maximumTokensPerChunk,
                    sampleToMillis(offset), sampleToMillis(count)));
        }
        return combine(stitcher.acceptedWindows());
    }

    /** Use a small encoder window for short recordings and fixed 30-second windows for longer audio. */
    public TranscriptionResult transcribeShortRecordingWithTimestamps(float[] samples,
                                                                       int maximumTokensPerChunk)
            throws IOException {
        Objects.requireNonNull(samples, "samples");
        validateTokenLimit(maximumTokensPerChunk);
        List<TranscriptionResult> parts = new ArrayList<>();
        WhisperTimestampDecoder timestampDecoder = newTimestampDecoder();
        int samplesPerPart = WhisperLogMelSpectrogram.SAMPLE_RATE * 6;
        if (samples.length > samplesPerPart) {
            return transcribeWithTimestamps(samples, maximumTokensPerChunk);
        }
        for (int offset = 0; offset < samples.length; offset += samplesPerPart) {
            int sampleCount = Math.min(samplesPerPart, samples.length - offset);
            float[] chunk = java.util.Arrays.copyOfRange(samples, offset, offset + sampleCount);
            float[][] fullMel = frontend.compute(chunk);
            int frames = Math.min(WhisperLogMelSpectrogram.FRAMES_PER_CHUNK,
                    Math.max(40, (sampleCount + WhisperLogMelSpectrogram.HOP_LENGTH - 1)
                            / WhisperLogMelSpectrogram.HOP_LENGTH + 60));
            float[][] shortMel = new float[config.melBands()][frames];
            for (int band = 0; band < config.melBands(); band++) {
                System.arraycopy(fullMel[band], 0, shortMel[band], 0, frames);
            }
            parts.add(timestampDecoder.transcribe(shortMel, maximumTokensPerChunk,
                    sampleToMillis(offset), sampleToMillis(sampleCount)));
        }
        return combine(parts);
    }

    /** Decode prepared log-Mel features and model-emitted timestamp tokens. */
    public TranscriptionResult transcribeMelWithTimestamps(float[][] mel, int maximumTokens,
                                                           long windowStartMillis,
                                                           long windowDurationMillis)
            throws IOException {
        validateTokenLimit(maximumTokens);
        if (windowStartMillis < 0 || windowDurationMillis < 0 || windowDurationMillis > 30000) {
            throw new IllegalArgumentException("Invalid Whisper audio window");
        }
        return newTimestampDecoder().transcribe(mel, maximumTokens,
                windowStartMillis, windowDurationMillis);
    }

    private WhisperTimestampDecoder newTimestampDecoder() {
        return new WhisperTimestampDecoder(encoder, decoder, tokenizer, config);
    }

    private void validateTokenLimit(int maximumTokens) {
        if (maximumTokens < 1 || maximumTokens > config.textContext() - 4) {
            throw new IllegalArgumentException("Invalid Whisper token limit");
        }
    }

    private static long sampleToMillis(int samples) {
        return (long) samples * 1000 / WhisperLogMelSpectrogram.SAMPLE_RATE;
    }

    private static TranscriptionResult combine(List<TranscriptionResult> parts) {
        StringBuilder text = new StringBuilder();
        StringBuilder unaligned = new StringBuilder();
        List<TranscriptionSegment> segments = new ArrayList<>();
        for (TranscriptionResult part : parts) {
            appendText(text, part.getText());
            appendText(unaligned, part.getUnalignedText());
            segments.addAll(part.getSegments());
        }
        return new TranscriptionResult(text.toString(), unaligned.toString(), segments);
    }

    private static void appendText(StringBuilder output, String next) {
        if (next.isBlank()) {
            return;
        }
        if (output.length() > 0) {
            output.append(' ');
        }
        output.append(next.strip());
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
