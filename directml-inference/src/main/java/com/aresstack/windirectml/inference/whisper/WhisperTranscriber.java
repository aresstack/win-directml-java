package com.aresstack.windirectml.inference.whisper;

import com.aresstack.windirectml.inference.whisper.audio.WhisperLogMelSpectrogram;
import com.aresstack.windirectml.inference.whisper.audio.WhisperWavAudioLoader;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Execute multilingual Whisper speech-to-text completely inside the Java 21 JVM.
 * Keep the transcription use case independent of the file decoder and backend.
 *
 * <p>Every entry point shares one guarded decoding path: windows of at most thirty
 * seconds, padded to the model's native input length, decoded with timestamp tokens,
 * temperature fallback, compression-ratio, log-probability and no-speech checks, and
 * joined at model-confirmed segment boundaries. Plain-text methods return the same
 * transcript without timestamps. {@link WhisperDecodingOptions#defaults()} is tuned
 * for dictation and needs no further configuration.
 */
public final class WhisperTranscriber {
    private static final int CHUNK_LENGTH = WhisperLogMelSpectrogram.SAMPLES_PER_CHUNK;

    private final WhisperModelConfig config;
    private final WhisperAudioEncoder encoder;
    private final WhisperTextDecoder decoder;
    private final WhisperTokenizer tokenizer;
    private final WhisperLogMelSpectrogram frontend;
    private final WhisperWindowDecoder windowDecoder;

    public WhisperTranscriber(Path modelDirectory) throws IOException {
        Objects.requireNonNull(modelDirectory, "modelDirectory");
        config = WhisperModelConfig.load(modelDirectory.resolve("config.json"));
        WhisperCheckpoint checkpoint = WhisperCheckpoint.open(modelDirectory.resolve("model.safetensors"));
        encoder = new WhisperAudioEncoder(checkpoint, config);
        decoder = new WhisperTextDecoder(checkpoint, config);
        tokenizer = new WhisperTokenizer(modelDirectory);
        frontend = new WhisperLogMelSpectrogram(config.melBands());
        windowDecoder = new WhisperWindowDecoder(encoder, decoder, tokenizer, config,
                WhisperGenerationConfig.load(modelDirectory, config));
    }

    /**
     * Transcribe a window-readable 16-kHz mono source of any length with the guarded decoder.
     * Report every committed window on the decoding thread with absolute timestamps.
     * Stop with an {@link InterruptedIOException} when the decoding thread is interrupted.
     */
    public TranscriptionResult transcribe(WhisperAudioSource source, WhisperDecodingOptions options,
                                          Consumer<TranscriptionResult> onProgress) throws IOException {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(options, "options");
        Objects.requireNonNull(onProgress, "onProgress");
        validateTokenLimit(options.maximumTokensPerWindow());
        WhisperTimestampWindowStitcher stitcher = new WhisperTimestampWindowStitcher(source.sampleCount());
        while (stitcher.hasNextWindow()) {
            onProgress.accept(decodeNextWindow(stitcher, source, options));
        }
        return combine(stitcher.acceptedWindows());
    }

    /**
     * Transcribe a recording that is still growing. Decode completed 30-second windows with
     * {@link WhisperIncrementalTranscription#decodeAvailable()} while recording continues, and the
     * rest with {@link WhisperIncrementalTranscription#finish()} once it has stopped. The windows,
     * safeguards and stitching rules are those of the batch path, so both produce the same text.
     *
     * @param growingSource a source whose {@code sampleCount()} grows while recording; every reported
     *                      sample must remain readable and unchanged
     */
    public WhisperIncrementalTranscription transcribeIncrementally(WhisperAudioSource growingSource,
                                                                   WhisperDecodingOptions options,
                                                                   Consumer<TranscriptionResult> onCommitted) {
        Objects.requireNonNull(growingSource, "growingSource");
        Objects.requireNonNull(options, "options");
        Objects.requireNonNull(onCommitted, "onCommitted");
        validateTokenLimit(options.maximumTokensPerWindow());
        return new WhisperIncrementalTranscription(this, growingSource, options, onCommitted);
    }

    /** Decode and commit the window at the stitcher's next position. */
    TranscriptionResult decodeNextWindow(WhisperTimestampWindowStitcher stitcher, WhisperAudioSource source,
                                         WhisperDecodingOptions options) throws IOException {
        if (Thread.currentThread().isInterrupted()) {
            throw new InterruptedIOException("Whisper transcription was interrupted");
        }
        long offset = stitcher.nextStartSample();
        int count = stitcher.nextWindowSamples();
        float[] samples = source.readSamples(offset, count);
        if (samples.length != count) {
            throw new IOException("Audio source did not return the requested sample window");
        }
        WhisperWindowDecoder.WindowDecoding window = windowDecoder.decode(
                melForWindow(samples, options), sampleToMillis(offset), sampleToMillis(count),
                options, true);
        return stitcher.accept(window.result, window.trailingOpenText);
    }

    /** Transcribe already decoded 16-kHz mono PCM with the guarded decoder. */
    public TranscriptionResult transcribe(float[] samples, WhisperDecodingOptions options,
                                          Consumer<TranscriptionResult> onProgress) throws IOException {
        return transcribe(arraySource(samples), options, onProgress);
    }

    /** Decode a WAV recording into 16 kHz mono float PCM, then transcribe German speech. */
    public String transcribe(Path wavFile, int maximumTokensPerChunk) throws IOException {
        return transcribe(new WhisperWavAudioLoader().load(wavFile), maximumTokensPerChunk);
    }

    /** Accept already decoded 16-kHz mono float PCM from another audio pipeline. */
    public String transcribe(float[] audioSamples, int maximumTokensPerChunk) throws IOException {
        return transcribe(audioSamples, options(maximumTokensPerChunk), ignored -> { }).getText();
    }

    /**
     * Transcribe a short, already decoded 16-kHz mono microphone recording.
     *
     * @deprecated Since 0.3.2 short recordings use the same guarded thirty-second path as
     * every other entry point; use {@link #transcribe(float[], WhisperDecodingOptions, Consumer)}.
     */
    @Deprecated
    public String transcribeShortRecording(float[] samples, int maximumTokensPerChunk) throws IOException {
        return transcribe(samples, maximumTokensPerChunk);
    }

    /**
     * Transcribe a window-readable source as plain text.
     *
     * @deprecated Since 0.3.2 use {@link #transcribe(WhisperAudioSource, WhisperDecodingOptions, Consumer)}.
     */
    @Deprecated
    public String transcribeShortRecording(WhisperAudioSource source, int maximumTokensPerChunk)
            throws IOException {
        return transcribe(source, options(maximumTokensPerChunk), ignored -> { }).getText();
    }

    /** Return Whisper-generated speech intervals for a WAV recording. */
    public TranscriptionResult transcribeWithTimestamps(Path wavFile, int maximumTokensPerChunk)
            throws IOException {
        return transcribeWithTimestamps(wavFile, maximumTokensPerChunk, chunk -> { });
    }

    /** Report finalized timestamped segments as each WAV window finishes decoding. */
    public TranscriptionResult transcribeWithTimestamps(Path wavFile, int maximumTokensPerChunk,
                                                       Consumer<TranscriptionResult> onProgress)
            throws IOException {
        Objects.requireNonNull(onProgress, "onProgress");
        return transcribeWithTimestamps(new WhisperWavAudioLoader().load(wavFile),
                maximumTokensPerChunk, onProgress);
    }

    /** Transcribe 16-kHz mono PCM using timestamp-driven 30-second windows. */
    public TranscriptionResult transcribeWithTimestamps(float[] samples, int maximumTokensPerChunk)
            throws IOException {
        return transcribeWithTimestamps(samples, maximumTokensPerChunk, chunk -> { });
    }

    /** Report only committed segments after each window, with absolute timestamps. */
    public TranscriptionResult transcribeWithTimestamps(float[] samples, int maximumTokensPerChunk,
                                                       Consumer<TranscriptionResult> onProgress)
            throws IOException {
        return transcribe(arraySource(samples), options(maximumTokensPerChunk), onProgress);
    }

    /** Transcribe a window-readable audio source without loading the complete recording. */
    public TranscriptionResult transcribeWithTimestamps(WhisperAudioSource source,
                                                        int maximumTokensPerChunk,
                                                        Consumer<TranscriptionResult> onProgress)
            throws IOException {
        return transcribe(source, options(maximumTokensPerChunk), onProgress);
    }

    /**
     * @deprecated Since 0.3.2 short recordings use the guarded thirty-second path;
     * use {@link #transcribe(float[], WhisperDecodingOptions, Consumer)}.
     */
    @Deprecated
    public TranscriptionResult transcribeShortRecordingWithTimestamps(float[] samples,
                                                                       int maximumTokensPerChunk)
            throws IOException {
        return transcribeWithTimestamps(samples, maximumTokensPerChunk);
    }

    /**
     * @deprecated Since 0.3.2 short recordings use the guarded thirty-second path;
     * use {@link #transcribe(float[], WhisperDecodingOptions, Consumer)}.
     */
    @Deprecated
    public TranscriptionResult transcribeShortRecordingWithTimestamps(float[] samples,
                                                                       int maximumTokensPerChunk,
                                                                       Consumer<TranscriptionResult> onProgress)
            throws IOException {
        return transcribeWithTimestamps(samples, maximumTokensPerChunk, onProgress);
    }

    /** Decode prepared log-Mel features of one window with the guarded timestamp decoder. */
    public TranscriptionResult transcribeMelWithTimestamps(float[][] mel, int maximumTokens,
                                                           long windowStartMillis,
                                                           long windowDurationMillis)
            throws IOException {
        validateTokenLimit(maximumTokens);
        if (windowStartMillis < 0 || windowDurationMillis < 0 || windowDurationMillis > 30000) {
            throw new IllegalArgumentException("Invalid Whisper audio window");
        }
        return windowDecoder.decode(mel, windowStartMillis, windowDurationMillis,
                options(maximumTokens), true).result;
    }

    /**
     * Transcribe a prepared mel spectrogram without timestamp tokens, using the same
     * temperature fallback and quality checks. Smaller frame counts serve focused tests.
     */
    public String transcribeMel(float[][] mel, int maximumTokens) throws IOException {
        validateTokenLimit(maximumTokens);
        long durationMillis = Math.min(30000L, mel[0].length * 10L);
        return windowDecoder.decode(mel, 0, durationMillis, options(maximumTokens), false)
                .result.getText();
    }

    /**
     * Compute the encoder input of one window. Full windows always use 3000 frames;
     * shorter windows are padded to that length unless the options request the
     * speech-length encoder input, which keeps sixty frames of padding after the audio.
     */
    private float[][] melForWindow(float[] samples, WhisperDecodingOptions options) {
        float[][] fullMel = frontend.compute(samples);
        if (samples.length == CHUNK_LENGTH || options.fullWindowEncoderInput()) {
            return fullMel;
        }
        int frames = Math.min(WhisperLogMelSpectrogram.FRAMES_PER_CHUNK,
                Math.max(40, (samples.length + WhisperLogMelSpectrogram.HOP_LENGTH - 1)
                        / WhisperLogMelSpectrogram.HOP_LENGTH + 60));
        float[][] shortMel = new float[config.melBands()][frames];
        for (int band = 0; band < config.melBands(); band++) {
            System.arraycopy(fullMel[band], 0, shortMel[band], 0, frames);
        }
        return shortMel;
    }

    private WhisperDecodingOptions options(int maximumTokens) {
        validateTokenLimit(maximumTokens);
        return WhisperDecodingOptions.defaults().withMaximumTokensPerWindow(maximumTokens);
    }

    private void validateTokenLimit(int maximumTokens) {
        if (maximumTokens < 1 || maximumTokens > config.textContext() - 4) {
            throw new IllegalArgumentException("Invalid Whisper token limit");
        }
    }

    private static WhisperAudioSource arraySource(float[] samples) {
        Objects.requireNonNull(samples, "samples");
        return new WhisperAudioSource() {
            @Override
            public long sampleCount() {
                return samples.length;
            }

            @Override
            public float[] readSamples(long startSample, int sampleCount) {
                int offset = Math.toIntExact(startSample);
                return java.util.Arrays.copyOfRange(samples, offset, offset + sampleCount);
            }
        };
    }

    private static long sampleToMillis(long samples) {
        return samples * 1000 / WhisperLogMelSpectrogram.SAMPLE_RATE;
    }

    static TranscriptionResult combine(List<TranscriptionResult> parts) {
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
}
