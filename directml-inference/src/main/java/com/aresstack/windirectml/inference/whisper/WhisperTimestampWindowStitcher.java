package com.aresstack.windirectml.inference.whisper;

import com.aresstack.windirectml.inference.whisper.audio.WhisperLogMelSpectrogram;

import java.util.ArrayList;
import java.util.List;

/**
 * Join timestamped Whisper windows of at most thirty seconds without losing speech.
 *
 * <p>Before the final window, only closed segments are committed. The next window starts
 * where the last committed segment ends, so text that ran into the window boundary and
 * any audio Whisper did not transcribe are decoded again with full context. A closed
 * segment touching the boundary is treated as cut off and decoded again from its start.
 * Only a window without any usable timestamp, or one that would advance by less than a
 * second, is committed completely to guarantee forward progress.
 */
final class WhisperTimestampWindowStitcher {
    private static final int SAMPLE_RATE = WhisperLogMelSpectrogram.SAMPLE_RATE;
    private static final int WINDOW_SAMPLES = WhisperLogMelSpectrogram.SAMPLES_PER_CHUNK;
    private static final int MINIMUM_ADVANCE_SAMPLES = SAMPLE_RATE;
    private static final int BOUNDARY_MARGIN_SAMPLES = SAMPLE_RATE / 2;

    /** The recording length; {@link Long#MAX_VALUE} while a growing recording is still open. */
    private long totalSamples;
    private final List<TranscriptionResult> acceptedWindows = new ArrayList<>();
    private long nextStartSample;

    WhisperTimestampWindowStitcher(long totalSamples) {
        if (totalSamples < 0) {
            throw new IllegalArgumentException("Audio length must not be negative");
        }
        this.totalSamples = totalSamples;
    }

    /** Stitch a recording that is still growing; every window is non-final until {@link #complete}. */
    static WhisperTimestampWindowStitcher open() {
        return new WhisperTimestampWindowStitcher(Long.MAX_VALUE);
    }

    /** Fix the final length of a growing recording; the remaining windows use final-window rules. */
    void complete(long finalSamples) {
        if (finalSamples < nextStartSample) {
            throw new IllegalArgumentException("The recording cannot end before committed audio");
        }
        totalSamples = finalSamples;
    }

    boolean hasNextWindow() {
        return nextStartSample < totalSamples;
    }

    long nextStartSample() {
        return nextStartSample;
    }

    int nextWindowSamples() {
        return (int) Math.min(WINDOW_SAMPLES, totalSamples - nextStartSample);
    }

    /** Accept a window whose last segment is known to be closed. */
    TranscriptionResult accept(TranscriptionResult recognition) {
        return accept(recognition, false);
    }

    /**
     * Commit the reliable part of a decoded window and choose the next window start.
     *
     * @param trailingOpenText whether the decoder stopped inside a segment
     */
    TranscriptionResult accept(TranscriptionResult recognition, boolean trailingOpenText) {
        long windowStart = nextStartSample;
        long windowEnd = windowStart + nextWindowSamples();
        List<TranscriptionSegment> segments = recognition.getSegments();
        if (windowEnd == totalSamples || segments.isEmpty()) {
            return commitWindow(recognition, windowEnd);
        }

        int committed = segments.size();
        TranscriptionSegment last = segments.get(committed - 1);
        long resumeSample = toSamples(last.getEndMillis());
        if (!trailingOpenText && resumeSample >= windowEnd - BOUNDARY_MARGIN_SAMPLES) {
            committed--;
            resumeSample = toSamples(last.getStartMillis());
        }
        if (resumeSample < windowStart + MINIMUM_ADVANCE_SAMPLES || resumeSample >= windowEnd) {
            return commitWindow(recognition, windowEnd);
        }

        List<TranscriptionSegment> confirmed = new ArrayList<>(segments.subList(0, committed));
        TranscriptionResult accepted = new TranscriptionResult(
                textOf(confirmed), recognition.getUnalignedText(), confirmed);
        acceptedWindows.add(accepted);
        nextStartSample = resumeSample;
        return accepted;
    }

    List<TranscriptionResult> acceptedWindows() {
        return acceptedWindows;
    }

    private TranscriptionResult commitWindow(TranscriptionResult recognition, long windowEnd) {
        acceptedWindows.add(recognition);
        nextStartSample = windowEnd;
        return recognition;
    }

    private static long toSamples(long millis) {
        return millis * SAMPLE_RATE / 1000L;
    }

    private static String textOf(List<TranscriptionSegment> segments) {
        StringBuilder text = new StringBuilder();
        for (TranscriptionSegment segment : segments) {
            if (text.length() > 0) {
                text.append(' ');
            }
            text.append(segment.getText());
        }
        return text.toString();
    }
}
