package com.aresstack.windirectml.inference.whisper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Locate speech activity and split long recordings at acoustic pauses.
 * Preserve sample offsets rather than inventing timestamps from window boundaries.
 */
final class WhisperSpeechWindow {
    private static final int SAMPLE_RATE = 16000;
    private static final int FRAME_SAMPLES = SAMPLE_RATE / 50;
    private static final int CONTEXT_PADDING_SAMPLES = SAMPLE_RATE / 10;
    private static final int MINIMUM_GROUP_SAMPLES = SAMPLE_RATE * 3;
    private static final int PAUSE_SAMPLES = SAMPLE_RATE * 45 / 100;
    private static final double MINIMUM_ACTIVITY_RMS = 0.0008;

    private final int firstSample;
    private final int endSample;

    private WhisperSpeechWindow(int firstSample, int endSample) {
        this.firstSample = firstSample;
        this.endSample = endSample;
    }

    static List<WhisperSpeechWindow> split(float[] audio, int maximumSamplesPerWindow) {
        Objects.requireNonNull(audio, "audio");
        if (maximumSamplesPerWindow < SAMPLE_RATE) {
            throw new IllegalArgumentException("Whisper requires windows of at least one second");
        }
        if (audio.length == 0) {
            return Collections.emptyList();
        }
        double peakRms = findPeakRms(audio, 0, audio.length);
        if (peakRms < MINIMUM_ACTIVITY_RMS) {
            return Collections.emptyList();
        }
        double threshold = Math.max(MINIMUM_ACTIVITY_RMS, peakRms * 0.08);
        List<WhisperSpeechWindow> spans = findSpeechSpans(audio, threshold);
        return groupIntoWindows(spans, audio.length, maximumSamplesPerWindow);
    }

    static Optional<WhisperSpeechWindow> detect(float[] audio, int start, int sampleCount) {
        Objects.requireNonNull(audio, "audio");
        if (start < 0 || sampleCount < 0 || (long) start + sampleCount > audio.length) {
            throw new IllegalArgumentException("Invalid audio window");
        }
        if (sampleCount == 0) {
            return Optional.empty();
        }
        double peak = findPeakRms(audio, start, start + sampleCount);
        if (peak < MINIMUM_ACTIVITY_RMS) {
            return Optional.empty();
        }
        double threshold = Math.max(MINIMUM_ACTIVITY_RMS, peak * 0.08);
        int firstActive = start + sampleCount;
        int lastActive = start;
        for (int frame = start; frame < start + sampleCount; frame += FRAME_SAMPLES) {
            int frameEnd = Math.min(start + sampleCount, frame + FRAME_SAMPLES);
            if (rootMeanSquare(audio, frame, frameEnd) >= threshold) {
                firstActive = Math.min(firstActive, frame);
                lastActive = frameEnd;
            }
        }
        if (firstActive == start + sampleCount) {
            return Optional.empty();
        }
        return Optional.of(new WhisperSpeechWindow(
                Math.max(start, firstActive - CONTEXT_PADDING_SAMPLES),
                Math.min(start + sampleCount, lastActive + CONTEXT_PADDING_SAMPLES)));
    }

    int firstSample() {
        return firstSample;
    }

    int sampleCount() {
        return endSample - firstSample;
    }

    private static List<WhisperSpeechWindow> findSpeechSpans(float[] audio, double threshold) {
        List<WhisperSpeechWindow> spans = new ArrayList<>();
        int beginning = -1;
        int lastActive = -1;
        for (int frame = 0; frame < audio.length; frame += FRAME_SAMPLES) {
            int end = Math.min(audio.length, frame + FRAME_SAMPLES);
            if (rootMeanSquare(audio, frame, end) < threshold) {
                continue;
            }
            if (beginning >= 0 && frame - lastActive >= PAUSE_SAMPLES) {
                spans.add(new WhisperSpeechWindow(beginning, lastActive));
                beginning = -1;
            }
            if (beginning < 0) {
                beginning = frame;
            }
            lastActive = end;
        }
        if (beginning >= 0) {
            spans.add(new WhisperSpeechWindow(beginning, lastActive));
        }
        return spans;
    }

    private static List<WhisperSpeechWindow> groupIntoWindows(
            List<WhisperSpeechWindow> spans, int audioLength, int maximumSamplesPerWindow) {
        List<WhisperSpeechWindow> windows = new ArrayList<>();
        int first = -1;
        int end = -1;
        for (WhisperSpeechWindow span : spans) {
            if (first >= 0 && (span.endSample - first > maximumSamplesPerWindow
                    || end - first >= MINIMUM_GROUP_SAMPLES)) {
                appendWindow(windows, first, end, audioLength, maximumSamplesPerWindow);
                first = -1;
            }
            if (first < 0) {
                first = span.firstSample;
            }
            end = span.endSample;
        }
        if (first >= 0) {
            appendWindow(windows, first, end, audioLength, maximumSamplesPerWindow);
        }
        return windows;
    }

    private static void appendWindow(List<WhisperSpeechWindow> windows, int start, int end,
                                     int audioLength, int maximumSamplesPerWindow) {
        int cursor = Math.max(0, start - CONTEXT_PADDING_SAMPLES);
        int limit = Math.min(audioLength, end + CONTEXT_PADDING_SAMPLES);
        while (cursor < limit) {
            int next = (int) Math.min((long) limit, (long) cursor + maximumSamplesPerWindow);
            windows.add(new WhisperSpeechWindow(cursor, next));
            cursor = next;
        }
    }

    private static double findPeakRms(float[] audio, int start, int end) {
        double peak = 0;
        for (int frame = start; frame < end; frame += FRAME_SAMPLES) {
            peak = Math.max(peak, rootMeanSquare(audio, frame, Math.min(end, frame + FRAME_SAMPLES)));
        }
        return peak;
    }

    private static double rootMeanSquare(float[] audio, int begin, int end) {
        double sum = 0;
        for (int index = begin; index < end; index++) {
            double sample = audio[index];
            sum += sample * sample;
        }
        return Math.sqrt(sum / (end - begin));
    }
}
