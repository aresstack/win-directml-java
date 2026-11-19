package com.aresstack.windirectml.workbench.whisper;

import com.aresstack.audio.dsp.SpeechActivityMetadata;
import com.aresstack.audio.dsp.VoiceActivityDetector;
import com.aresstack.audio.dsp.VoiceActivityDetectorSettings;
import com.aresstack.audio.dsp.VoiceActivityDetectorState;
import com.aresstack.windirectml.inference.whisper.WhisperAudioSource;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

/**
 * Find speech regions using the deterministic Java VAD from Ask AI.
 * Read small chunks to preserve the bounded microphone recording memory model.
 */
final class SpeechRegionDetector {
    private static final int SAMPLE_RATE = 16000;
    private static final int FRAME_SAMPLES = SAMPLE_RATE * 20 / 1000;
    private static final int FRAMES_PER_READ = 100;

    private final VoiceActivityDetector detector = new VoiceActivityDetector();
    private final VoiceActivityDetectorSettings settings =
            new VoiceActivityDetectorSettings(0.65d, 0.48d, 20,
                    40.0d, 140.0d, 120.0d, 60.0d, 120.0d, 0.05d, false);

    List<SpeechInterval> detect(WhisperAudioSource source) throws IOException {
        Objects.requireNonNull(source, "source");
        if (source.sampleCount() == 0) {
            return java.util.Collections.emptyList();
        }
        WhisperAudioSource filteredSource = new AskAiProcessedSpeechSource(source,
                new SpeechInterval(0, source.sampleCount()), AskAiProcessedSpeechSource.vadProfile());
        SpeechIntervalCollector collector = new SpeechIntervalCollector();
        VoiceActivityDetectorState state = new VoiceActivityDetectorState();
        short[] frame = new short[FRAME_SAMPLES];
        initializeNoiseFloor(filteredSource, state);

        long start = 0;
        while (start < filteredSource.sampleCount()) {
            int count = (int) Math.min(FRAMES_PER_READ * FRAME_SAMPLES,
                    filteredSource.sampleCount() - start);
            float[] audio = filteredSource.readSamples(start, count);
            if (audio.length != count) {
                throw new IOException("Audio source returned an incomplete VAD interval");
            }
            for (int offset = 0; offset < count; offset += FRAME_SAMPLES) {
                int available = Math.min(FRAME_SAMPLES, count - offset);
                java.util.Arrays.fill(frame, (short) 0);
                for (int i = 0; i < available; i++) {
                    frame[i] = toPcm16(audio[offset + i]);
                }
                SpeechActivityMetadata activity = detector.analyzeFrame(frame, 0, FRAME_SAMPLES, 1,
                        SAMPLE_RATE, settings, state);
                long frameStart = start + offset;
                collector.accept(frameStart, frameStart + available,
                        activity.isSpeechActive() ? 1.0d : 0.0d);
            }
            start += count;
        }
        return collector.finish(source.sampleCount());
    }

    private void initializeNoiseFloor(WhisperAudioSource source, VoiceActivityDetectorState state)
            throws IOException {
        int sampleCount = (int) Math.min(FRAME_SAMPLES * FRAMES_PER_READ, source.sampleCount());
        float[] calibrationAudio = source.readSamples(0, sampleCount);
        short[] quietest = new short[FRAME_SAMPLES];
        double minimumEnergy = Double.POSITIVE_INFINITY;
        for (int offset = 0; offset < calibrationAudio.length; offset += FRAME_SAMPLES) {
            short[] candidate = new short[FRAME_SAMPLES];
            double energy = 0;
            int end = Math.min(FRAME_SAMPLES, calibrationAudio.length - offset);
            for (int i = 0; i < end; i++) {
                candidate[i] = toPcm16(calibrationAudio[offset + i]);
                energy += (double) candidate[i] * candidate[i];
            }
            if (energy < minimumEnergy) {
                minimumEnergy = energy;
                quietest = candidate;
            }
        }
        // Calibrate from the quietest observed frame, not a word spoken at sample zero.
        detector.analyzeFrame(quietest, 0, FRAME_SAMPLES, 1, SAMPLE_RATE, settings, state);
    }

    static short toPcm16(float sample) {
        return (short) Math.max(Short.MIN_VALUE,
                Math.min(Short.MAX_VALUE, Math.round(sample * 32768.0f)));
    }
}
