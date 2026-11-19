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

    private final int totalSamples;
    private final List<TranscriptionResult> acceptedWindows = new ArrayList<>();
    private int nextStartSample;

    WhisperTimestampWindowStitcher(int totalSamples) {
        if (totalSamples < 0) {
            throw new IllegalArgumentException("Audio length must not be negative");
        }
        this.totalSamples = totalSamples;
    }

    boolean hasNextWindow() {
        return nextStartSample < totalSamples;
    }

    int nextStartSample() {
        return nextStartSample;
    }

    void accept(TranscriptionResult recognition) {
        int windowEndSample = nextStartSample + Math.min(WINDOW_SAMPLES, totalSamples - nextStartSample);
        if (windowEndSample == totalSamples) {
            acceptCompleteWindow(recognition, windowEndSample);
            return;
        }

        List<TranscriptionSegment> segments = recognition.getSegments();
        if (segments.size() < 2) {
            acceptCompleteWindow(recognition, windowEndSample);
            return;
        }

        long nextStartMillis = segments.get(segments.size() - 1).getStartMillis();
        long nextStartInSamples = nextStartMillis * SAMPLE_RATE / 1000L;
        if (nextStartInSamples <= nextStartSample + MINIMUM_ADVANCE_SAMPLES
                || nextStartInSamples >= windowEndSample) {
            acceptCompleteWindow(recognition, windowEndSample);
            return;
        }

        List<TranscriptionSegment> confirmed = new ArrayList<>(
                segments.subList(0, segments.size() - 1));
        acceptedWindows.add(new TranscriptionResult(
                textOf(confirmed), recognition.getUnalignedText(), confirmed));
        nextStartSample = (int) nextStartInSamples;
    }

    List<TranscriptionResult> acceptedWindows() {
        return acceptedWindows;
    }

    private void acceptCompleteWindow(TranscriptionResult recognition, int windowEndSample) {
        acceptedWindows.add(recognition);
        nextStartSample = windowEndSample;
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
