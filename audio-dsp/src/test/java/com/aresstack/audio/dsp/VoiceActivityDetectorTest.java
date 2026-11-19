package com.aresstack.audio.dsp;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The adaptive voice-activity detector on synthetic, reproducible speech-LIKE test signals (not real
 * human speech): detection, stabilization, adaptive noise floor and finiteness.
 */
public class VoiceActivityDetectorTest {

    private static final int RATE = 16000;

    // ------------------------------------------------------------------ detection

    @Test
    public void digitalSilenceIsNeverActive() {
        List<SpeechActivityMetadata> track = run(silence(RATE), settings(0.7d));
        assertEquals(0, activeCount(track));
    }

    @Test
    public void lowStationaryNoiseIsInactiveAfterLearning() {
        short[] signal = noise(RATE * 2, 120, 7);
        List<SpeechActivityMetadata> track = run(signal, settings(0.5d));
        // After the first ~10 frames the noise floor has learned the level; the tail must be silent.
        int activeInTail = 0;
        for (int i = 20; i < track.size(); i++) {
            if (track.get(i).isSpeechActive()) {
                activeInTail++;
            }
        }
        assertEquals("stationary noise must not be taken for speech once learned", 0, activeInTail);
    }

    @Test
    public void aStrongSpeechLikeSignalIsDetectedAsActive() {
        short[] signal = concat(noise(RATE, 100, 1), tone(RATE, 300.0d, 6000), noise(RATE / 2, 100, 2));
        List<SpeechActivityMetadata> track = run(signal, settings(0.5d));
        assertTrue("a loud speech-like burst should be detected", activeCount(track) > 10);
    }

    @Test
    public void aSingleImpulseIsNotAStableSpeechSegment() {
        short[] signal = noise(RATE, 100, 3);
        signal[RATE / 2] = 30000; // one loud sample
        List<SpeechActivityMetadata> track = run(signal, settings(0.5d));
        assertEquals("a lone impulse must not activate speech", 0, activeCount(track));
    }

    @Test
    public void aBurstShorterThanMinimumSpeechDurationDoesNotLatchActive() {
        // 40 ms burst with an 80 ms minimum speech duration must not activate.
        short[] burst = tone(RATE * 40 / 1000, 300.0d, 6000);
        short[] signal = concat(noise(RATE, 100, 4), burst, noise(RATE, 100, 5));
        List<SpeechActivityMetadata> track = run(signal, settings(0.5d));
        assertEquals(0, activeCount(track));
    }

    @Test
    public void hangoverHoldsActivityThenReleaseEndsIt() {
        short[] signal = concat(noise(RATE, 100, 6), tone(RATE, 300.0d, 6000), noise(RATE, 100, 7));
        VoiceActivityDetectorSettings s = new VoiceActivityDetectorSettings(
                0.5d, 0.5d, 20, 40.0d, 200.0d, 300.0d, 60.0d, 120.0d, 0.05d, false);
        List<SpeechActivityMetadata> track = run(signal, s);
        int burstEndFrame = (RATE + RATE) / (RATE * 20 / 1000); // end of the tone in frames
        assertTrue("still active shortly after the burst (hangover)", track.get(burstEndFrame + 2).isSpeechActive());
        assertTrue("inactive again well after the burst (release)",
                !track.get(track.size() - 1).isSpeechActive());
    }

    // ------------------------------------------------------------------ adaptive state

    @Test
    public void noiseFloorLearnsConstantBackgroundNoise() {
        short[] signal = noise(RATE * 2, 500, 8);
        List<SpeechActivityMetadata> track = run(signal, settings(0.5d));
        double first = track.get(0).getEstimatedNoiseLevelDb();
        SpeechActivityMetadata last = track.get(track.size() - 1);
        assertTrue("noise floor is finite", isFinite(last.getEstimatedNoiseLevelDb()));
        // It converges toward the measured level of the constant noise.
        assertTrue("floor tracks the noise level",
                Math.abs(last.getEstimatedNoiseLevelDb() - last.getMeasuredLevelDb()) < 4.0d);
        assertTrue(isFinite(first));
    }

    @Test
    public void graduallyRisingNoiseIsTracked() {
        // A gradual noise ramp (as the spec describes) stays below the speech threshold, so the floor tracks it.
        short[] ramp = risingNoise(RATE * 3, 120, 1800, 21);
        List<SpeechActivityMetadata> track = run(ramp, settings(0.5d));
        double early = track.get(10).getEstimatedNoiseLevelDb();
        double late = track.get(track.size() - 1).getEstimatedNoiseLevelDb();
        assertTrue("the noise floor rises with the gradually rising background: " + early + " -> " + late,
                late > early + 3.0d);
    }

    @Test
    public void speechDoesNotPullTheNoiseFloorUp() {
        short[] leadIn = noise(RATE, 120, 11);
        short[] burst = tone(RATE, 300.0d, 8000);
        List<SpeechActivityMetadata> track = run(concat(leadIn, burst), settings(0.5d));
        int leadFrames = RATE / (RATE * 20 / 1000);
        double floorBeforeBurst = track.get(leadFrames - 2).getEstimatedNoiseLevelDb();
        double floorDuringBurst = track.get(track.size() - 1).getEstimatedNoiseLevelDb();
        assertTrue("loud speech must not raise the noise floor much: before=" + floorBeforeBurst
                + " during=" + floorDuringBurst, floorDuringBurst < floorBeforeBurst + 3.0d);
    }

    @Test
    public void resetProducesReproducibleResultsAndRunsShareNoState() {
        short[] signal = concat(noise(RATE, 100, 12), tone(RATE, 300.0d, 6000));
        List<SpeechActivityMetadata> first = run(signal, settings(0.5d));
        List<SpeechActivityMetadata> second = run(signal, settings(0.5d));
        assertEquals(first.size(), second.size());
        for (int i = 0; i < first.size(); i++) {
            assertEquals("frame " + i, first.get(i).getSpeechProbability(),
                    second.get(i).getSpeechProbability(), 1.0e-12d);
            assertEquals(first.get(i).isSpeechActive(), second.get(i).isSpeechActive());
        }
    }

    // ------------------------------------------------------------------ parameters & metadata

    @Test
    public void higherSensitivityDetectsWeakerSignals() {
        short[] signal = concat(noise(RATE, 100, 13), tone(RATE, 300.0d, 320), noise(RATE / 2, 100, 14));
        int sensitive = activeCount(run(signal, settings(1.0d)));
        int strict = activeCount(run(signal, settings(0.0d)));
        assertTrue("higher sensitivity should detect a weak tone more: " + sensitive + " vs " + strict,
                sensitive > strict);
    }

    @Test
    public void higherProbabilityThresholdReducesActivations() {
        short[] signal = concat(noise(RATE, 100, 15), tone(RATE, 300.0d, 900), noise(RATE / 2, 100, 16));
        VoiceActivityDetectorSettings low = new VoiceActivityDetectorSettings(
                0.6d, 0.35d, 20, 40.0d, 200.0d, 100.0d, 60.0d, 120.0d, 0.05d, false);
        VoiceActivityDetectorSettings high = new VoiceActivityDetectorSettings(
                0.6d, 0.85d, 20, 40.0d, 200.0d, 100.0d, 60.0d, 120.0d, 0.05d, false);
        assertTrue(activeCount(run(signal, low)) >= activeCount(run(signal, high)));
    }

    @Test
    public void probabilityStaysInRangeAndLevelsStayFinite() {
        short[] signal = concat(noise(RATE, 300, 17), tone(RATE, 300.0d, 9000), silence(RATE / 2));
        for (SpeechActivityMetadata frame : run(signal, settings(0.5d))) {
            assertTrue(frame.getSpeechProbability() >= 0.0d && frame.getSpeechProbability() <= 1.0d);
            assertTrue(isFinite(frame.getEstimatedNoiseLevelDb()));
            assertTrue(isFinite(frame.getMeasuredLevelDb()));
        }
    }

    // ------------------------------------------------------------------ helpers

    private static VoiceActivityDetectorSettings settings(double sensitivity) {
        return new VoiceActivityDetectorSettings(sensitivity, 0.5d, 20, 40.0d, 200.0d, 100.0d,
                80.0d, 120.0d, 0.05d, false);
    }

    private static List<SpeechActivityMetadata> run(short[] mono, VoiceActivityDetectorSettings s) {
        VoiceActivityDetector detector = new VoiceActivityDetector();
        VoiceActivityDetectorState state = new VoiceActivityDetectorState();
        int frame = Math.max(1, Math.round(RATE * s.getFrameDurationMs() / 1000.0f));
        List<SpeechActivityMetadata> out = new ArrayList<SpeechActivityMetadata>();
        for (int start = 0; start < mono.length; start += frame) {
            int count = Math.min(frame, mono.length - start);
            out.add(detector.analyzeFrame(mono, start, count, 1, RATE, s, state));
        }
        return out;
    }

    private static int activeCount(List<SpeechActivityMetadata> track) {
        int count = 0;
        for (SpeechActivityMetadata frame : track) {
            if (frame.isSpeechActive()) {
                count++;
            }
        }
        return count;
    }

    private static short[] silence(int n) {
        return new short[n];
    }

    private static short[] noise(int n, int amplitude, int seed) {
        short[] out = new short[n];
        int state = seed * 2654435761L != 0 ? seed | 1 : 1;
        for (int i = 0; i < n; i++) {
            state = state * 1103515245 + 12345;
            int value = (state >>> 16) % (2 * amplitude + 1) - amplitude;
            out[i] = (short) value;
        }
        return out;
    }

    private static short[] risingNoise(int n, int startAmplitude, int endAmplitude, int seed) {
        short[] out = new short[n];
        int state = seed | 1;
        for (int i = 0; i < n; i++) {
            int amplitude = startAmplitude + (int) ((long) (endAmplitude - startAmplitude) * i / n);
            state = state * 1103515245 + 12345;
            out[i] = (short) ((state >>> 16) % (2 * amplitude + 1) - amplitude);
        }
        return out;
    }

    private static short[] tone(int n, double frequencyHz, int amplitude) {
        short[] out = new short[n];
        for (int i = 0; i < n; i++) {
            out[i] = (short) Math.round(amplitude * Math.sin(2.0d * Math.PI * frequencyHz * i / RATE));
        }
        return out;
    }

    private static short[] concat(short[]... parts) {
        int total = 0;
        for (short[] part : parts) {
            total += part.length;
        }
        short[] out = new short[total];
        int offset = 0;
        for (short[] part : parts) {
            System.arraycopy(part, 0, out, offset, part.length);
            offset += part.length;
        }
        return out;
    }

    private static boolean isFinite(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }
}
