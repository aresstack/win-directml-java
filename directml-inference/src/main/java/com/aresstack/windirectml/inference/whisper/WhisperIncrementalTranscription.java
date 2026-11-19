package com.aresstack.windirectml.inference.whisper;

import com.aresstack.windirectml.inference.whisper.audio.WhisperLogMelSpectrogram;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.util.function.Consumer;

/**
 * Transcribe a recording while it is still growing, window by window.
 *
 * <p>{@link #decodeAvailable()} decodes every complete 30-second window beyond the committed
 * position as a non-final window: only closed segments are committed and the next window starts
 * where the last committed segment ends. {@link #finish()} decodes the remaining audio with the
 * final-window rules once recording has stopped. Every committed window is reported to the
 * callback on the calling thread. Methods are synchronized, so a background worker can call
 * {@code decodeAvailable()} repeatedly and another thread can call {@code finish()} afterwards.
 */
public final class WhisperIncrementalTranscription {
    private static final int WINDOW_SAMPLES = WhisperLogMelSpectrogram.SAMPLES_PER_CHUNK;

    private final WhisperTranscriber transcriber;
    private final WhisperAudioSource source;
    private final WhisperDecodingOptions options;
    private final Consumer<TranscriptionResult> onCommitted;
    private final WhisperTimestampWindowStitcher stitcher = WhisperTimestampWindowStitcher.open();
    private boolean finished;

    WhisperIncrementalTranscription(WhisperTranscriber transcriber, WhisperAudioSource source,
                                    WhisperDecodingOptions options, Consumer<TranscriptionResult> onCommitted) {
        this.transcriber = transcriber;
        this.source = source;
        this.options = options;
        this.onCommitted = onCommitted;
    }

    /**
     * Decode all complete windows that are already recorded.
     *
     * @return whether at least one window was decoded
     */
    public synchronized boolean decodeAvailable() throws IOException {
        if (finished) {
            throw new IllegalStateException("The incremental transcription has already finished");
        }
        boolean decoded = false;
        while (source.sampleCount() - stitcher.nextStartSample() >= WINDOW_SAMPLES) {
            onCommitted.accept(transcriber.decodeNextWindow(stitcher, source, options));
            decoded = true;
        }
        return decoded;
    }

    /** The position up to which audio has been committed, in samples. */
    public synchronized long committedSamples() {
        return stitcher.nextStartSample();
    }

    /**
     * Decode the rest of the stopped recording and return the complete transcript. Calling it again
     * returns the same result.
     */
    public synchronized TranscriptionResult finish() throws IOException {
        if (!finished) {
            stitcher.complete(Math.max(stitcher.nextStartSample(), source.sampleCount()));
            while (stitcher.hasNextWindow()) {
                if (Thread.currentThread().isInterrupted()) {
                    throw new InterruptedIOException("Whisper transcription was interrupted");
                }
                onCommitted.accept(transcriber.decodeNextWindow(stitcher, source, options));
            }
            finished = true;
        }
        return WhisperTranscriber.combine(stitcher.acceptedWindows());
    }
}
