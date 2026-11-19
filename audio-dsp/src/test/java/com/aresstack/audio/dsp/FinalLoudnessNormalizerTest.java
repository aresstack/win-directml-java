package com.aresstack.audio.dsp;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** The Final Loudness Normalizer reaches the target level with one constant gain and honours the ceiling. */
public class FinalLoudnessNormalizerTest {

    private static final int RATE = 16000;

    private static FinalLoudnessNormalizerSettings rms(double targetDb) {
        return new FinalLoudnessNormalizerSettings(FinalLoudnessNormalizerSettings.Mode.TARGET_RMS,
                targetDb, 24.0d, 24.0d, -1.0d, true, true, true);
    }

    @Test
    public void raisesQuietSignalToTargetRms() {
        short[] s = tone(300.0d, 32000, 1000);
        new FinalLoudnessNormalizer(rms(-20.0d)).process(s, s.length);
        assertTrue("reached target: " + dbfs(rmsOf(s, 0, s.length)),
                within(dbfs(rmsOf(s, 0, s.length)), -20.0d, 2.0d));
    }

    @Test
    public void lowersLoudSignalToTargetRms() {
        short[] s = tone(300.0d, 32000, 25000);
        new FinalLoudnessNormalizer(rms(-20.0d)).process(s, s.length);
        assertTrue("reached target: " + dbfs(rmsOf(s, 0, s.length)),
                within(dbfs(rmsOf(s, 0, s.length)), -20.0d, 2.0d));
    }

    @Test
    public void respectsThePeakCeiling() {
        short[] s = tone(300.0d, 32000, 8000);
        FinalLoudnessNormalizerSettings peak = new FinalLoudnessNormalizerSettings(
                FinalLoudnessNormalizerSettings.Mode.PEAK, -1.0d, 24.0d, 24.0d, -1.0d, true, true, true);
        new FinalLoudnessNormalizer(peak).process(s, s.length);
        double ceilingAmp = Math.pow(10.0d, -1.0d / 20.0d) * 32768.0d;
        assertTrue("peak under ceiling: " + peakOf(s), peakOf(s) <= ceilingAmp + 2.0d);
        assertTrue("peak reached near ceiling", peakOf(s) > ceilingAmp - 400.0d);
    }

    @Test
    public void appliesOneConstantGainWithoutPumping() {
        short[] s = new short[32000];
        fill(s, 0, 16000, 300.0d, 2000);
        fill(s, 16000, 32000, 300.0d, 8000);
        double ratioBefore = rmsOf(s, 16000, 32000) / rmsOf(s, 0, 16000);
        new FinalLoudnessNormalizer(rms(-16.0d)).process(s, s.length);
        double ratioAfter = rmsOf(s, 16000, 32000) / rmsOf(s, 0, 16000);
        assertEquals("dynamics preserved (constant gain)", ratioBefore, ratioAfter, ratioBefore * 0.02d);
    }

    @Test
    public void doesNotAmplifyWhenAmplificationIsDisallowed() {
        short[] s = tone(300.0d, 16000, 1000);
        short[] copy = s.clone();
        FinalLoudnessNormalizerSettings noBoost = new FinalLoudnessNormalizerSettings(
                FinalLoudnessNormalizerSettings.Mode.TARGET_RMS, -20.0d, 24.0d, 24.0d, -1.0d, true, false, true);
        new FinalLoudnessNormalizer(noBoost).process(s, s.length);
        assertArrayEquals(copy, s);
    }

    private static void fill(short[] s, int from, int to, double freq, int amp) {
        for (int i = from; i < to; i++) {
            s[i] = (short) Math.round(amp * Math.sin(2.0d * Math.PI * freq * i / RATE));
        }
    }

    private static short[] tone(double freq, int n, int amp) {
        short[] s = new short[n];
        fill(s, 0, n, freq, amp);
        return s;
    }

    private static double rmsOf(short[] s, int from, int to) {
        long sum = 0;
        for (int i = from; i < to; i++) {
            sum += (long) s[i] * s[i];
        }
        return Math.sqrt((double) sum / (to - from));
    }

    private static double peakOf(short[] s) {
        double p = 0.0d;
        for (short v : s) {
            p = Math.max(p, Math.abs(v));
        }
        return p;
    }

    private static double dbfs(double rms) {
        return 20.0d * Math.log10(Math.max(rms, 1.0e-9d) / 32768.0d);
    }

    private static boolean within(double value, double target, double tol) {
        return Math.abs(value - target) <= tol;
    }
}
