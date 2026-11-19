package com.aresstack.windirectml.inference.whisper;

import com.aresstack.windirectml.inference.whisper.audio.WhisperLogMelSpectrogram;

import java.util.ArrayList;
import java.util.List;

/**
 * Join timestamped Whisper windows at the beginning of the last recognized segment.
 * Retain complete windows when their timestamps cannot guarantee forward progress.
 */
final class WhisperTimestampWindowStitcher {
    private static final int SAMPLE_RATE = WhisperLogMelSpectrogram.SAMPLE_RATE;
    private static final int WINDOW_SAMPLES = WhisperLogMelSpectrogram.SAMPLES_PER_CHUNK;
    private static final int MINIMUM_ADVANCE_SAMPLES = 5 * SAMPLE_RATE;

    private final long totalSamples;
    private final List<TranscriptionResult> acceptedWindows = new ArrayList<>();
    private long nextStartSample;

    WhisperTimestampWindowStitcher(long totalSamples) {
        if (totalSamples < 0) {
            throw new IllegalArgumentException("Audio length must not be negative");
        }
        this.totalSamples = totalSamples;
    }

    boolean hasNextWindow() {
        return nextStartSample < totalSamples;
    }

    long nextStartSample() {
        return nextStartSample;
    }

    TranscriptionResult accept(TranscriptionResult recognition) {
        long windowEndSample = nextStartSample + Math.min(WINDOW_SAMPLES, totalSamples - nextStartSample);
        if (windowEndSample == totalSamples) {
            return acceptCompleteWindow(recognition, windowEndSample);
        }

        List<TranscriptionSegment> segments = recognition.getSegments();
        if (segments.size() < 2) {
            return acceptCompleteWindow(recognition, windowEndSample);
        }

        long nextStartMillis = segments.get(segments.size() - 1).getStartMillis();
        long nextStartInSamples = nextStartMillis * SAMPLE_RATE / 1000L;
        if (nextStartInSamples <= nextStartSample + MINIMUM_ADVANCE_SAMPLES
                || nextStartInSamples >= windowEndSample) {
            return acceptCompleteWindow(recognition, windowEndSample);
        }

        List<TranscriptionSegment> confirmed = new ArrayList<>(
                segments.subList(0, segments.size() - 1));
        TranscriptionResult accepted = new TranscriptionResult(
                textOf(confirmed), recognition.getUnalignedText(), confirmed);
        acceptedWindows.add(accepted);
        nextStartSample = nextStartInSamples;
        return accepted;
    }

    List<TranscriptionResult> acceptedWindows() {
        return acceptedWindows;
    }

    private TranscriptionResult acceptCompleteWindow(TranscriptionResult recognition, long windowEndSample) {
        acceptedWindows.add(recognition);
        nextStartSample = windowEndSample;
        return recognition;
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
