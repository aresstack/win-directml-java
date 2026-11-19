package com.aresstack.windirectml.whisper.audio;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Convert model speech probabilities to padded regions on the original time axis.
 * Merge nearby regions and never collapse the source's sample positions.
 */
final class SpeechIntervalCollector {
    private static final int SAMPLE_RATE = 16000;
    private static final long MIN_SPEECH_SAMPLES = SAMPLE_RATE * 96L / 1000L;
    private static final long MIN_SILENCE_SAMPLES = SAMPLE_RATE * 350L / 1000L;
    private static final long PRE_ROLL_SAMPLES = SAMPLE_RATE * 200L / 1000L;
    private static final long POST_ROLL_SAMPLES = SAMPLE_RATE * 250L / 1000L;
    private static final double START_THRESHOLD = 0.50d;
    private static final double END_THRESHOLD = 0.35d;

    private final List<SpeechInterval> completed = new ArrayList<>();
    private long speechStart = -1;
    private long silenceStart = -1;

    void accept(long startSample, long endSample, double speechProbability) {
        if (speechStart < 0) {
            if (speechProbability >= START_THRESHOLD) {
                speechStart = startSample;
                silenceStart = -1;
            }
            return;
        }
        if (speechProbability >= END_THRESHOLD) {
            silenceStart = -1;
            return;
        }
        if (silenceStart < 0) {
            silenceStart = startSample;
        }
        if (endSample - silenceStart >= MIN_SILENCE_SAMPLES) {
            completeSpeech(silenceStart);
            speechStart = -1;
            silenceStart = -1;
        }
    }

    List<SpeechInterval> finish(long totalSamples) {
        if (speechStart >= 0) {
            completeSpeech(silenceStart >= 0 ? silenceStart : totalSamples);
        }
        List<SpeechInterval> padded = new ArrayList<>();
        for (SpeechInterval interval : completed) {
            long start = Math.max(0L, interval.startSample() - PRE_ROLL_SAMPLES);
            long end = Math.min(totalSamples, interval.endSample() + POST_ROLL_SAMPLES);
            if (!padded.isEmpty() && start <= padded.get(padded.size() - 1).endSample()) {
                SpeechInterval previous = padded.remove(padded.size() - 1);
                padded.add(new SpeechInterval(previous.startSample(),
                        Math.max(previous.endSample(), end)));
            } else if (end > start) {
                padded.add(new SpeechInterval(start, end));
            }
        }
        return Collections.unmodifiableList(padded);
    }

    private void completeSpeech(long endSample) {
        if (endSample - speechStart >= MIN_SPEECH_SAMPLES) {
            completed.add(new SpeechInterval(speechStart, endSample));
        }
    }
}
