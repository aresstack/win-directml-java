package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.dsp.DirectionEstimate;
import com.aresstack.audio.dsp.DirectionOfArrivalEstimator;
import com.aresstack.audio.dsp.DirectionTracker;
import com.aresstack.audio.dsp.MicrophoneArrayProfile;
import com.aresstack.audio.profile.AudioBlockType;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

/** Slice 12: direction-of-arrival estimation, smoothed tracking and moving-speaker beamforming. */
public class DirectionOfArrivalTest {

    private static final int RATE = 16000;
    private static final double C = 343000.0d;
    private static final PcmAudioFormat STEREO = new PcmAudioFormat(RATE, 2, 16);
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();
    private final MicrophoneArrayProfile array = MicrophoneArrayProfile.parse("a", "t", "-50,0,0; 50,0,0");

    @Test
    public void estimatesTheAzimuthOfASource() {
        short[] stereo = simulate(60.0d, noise(6000));
        DirectionEstimate est = DirectionOfArrivalEstimator.estimate(stereo, 0, 4000, 2, array, RATE, C, 32);
        assertTrue("azimuth near 60 deg: " + est.getAzimuthDeg(), Math.abs(est.getAzimuthDeg() - 60.0d) < 10.0d);
        assertTrue("confident", est.getConfidence() > 0.5d);
    }

    @Test
    public void trackerHoldsThroughALowConfidenceOutlier() {
        DirectionTracker tracker = new DirectionTracker(0.7d, 15.0d, 0.3d, 3, 90.0d, false);
        double a = 0.0d;
        for (int i = 0; i < 10; i++) {
            a = tracker.update(new DirectionEstimate(60.0d, 0.0d, 0.9d, true));
        }
        assertTrue("converged near 60: " + a, Math.abs(a - 60.0d) < 5.0d);
        double afterOutlier = tracker.update(new DirectionEstimate(10.0d, 0.0d, 0.05d, true)); // low confidence
        assertTrue("ignores low-confidence jump: " + afterOutlier, Math.abs(afterOutlier - 60.0d) < 5.0d);
    }

    @Test
    public void trackingBeamformerBeatsWrongFixedSteering() {
        short[] stereo = simulate(30.0d, noise(9000));
        short[] fixedWrong = com.aresstack.audio.dsp.DelayAndSumBeamformer.beamform(
                stereo.clone(), stereo.length, STEREO, array, 90.0d, 0.0d, C, null, 1.0d);

        AudioBuffer tracked = registry.createProcessor(AudioBlockType.DELAY_AND_SUM_BEAMFORMER)
                .process(new AudioBuffer(stereo.clone(), STEREO),
                        registry.defaultDefinition(AudioBlockType.DELAY_AND_SUM_BEAMFORMER, "b")
                                .withParameter("micPositionsMm", "-50,0,0; 50,0,0")
                                .withParameter("tracking", "true")
                                .withParameter("trackingBlockFrames", "256")
                                .withParameter("fallbackAzimuthDeg", "90")
                                .withParameter("minConfidence", "0.1"),
                        new AudioProcessingContext());
        assertTrue("tracking beam finds the source: track=" + energy(tracked.getSamples())
                        + " fixedWrong=" + energy(fixedWrong),
                energy(tracked.getSamples()) > energy(fixedWrong) * 1.1d);
    }

    /** Two mics on the x-axis; simulate a plane wave from the given azimuth over a padded source. */
    private static short[] simulate(double azimuthDeg, double[] source) {
        double az = Math.toRadians(azimuthDeg);
        double d0 = -50.0d * Math.cos(az) / C * RATE;
        double d1 = 50.0d * Math.cos(az) / C * RATE;
        int frames = source.length - 2000;
        short[] stereo = new short[frames * 2];
        for (int t = 0; t < frames; t++) {
            stereo[2 * t] = clamp(interp(source, t + 1000 + d0));
            stereo[2 * t + 1] = clamp(interp(source, t + 1000 + d1));
        }
        return stereo;
    }

    private static double[] noise(int n) {
        double[] s = new double[n];
        int state = 33377;
        for (int i = 0; i < n; i++) {
            state = state * 1103515245 + 12345;
            s[i] = (state >> 16) % 8000;
        }
        return s;
    }

    private static double interp(double[] s, double pos) {
        int i0 = (int) Math.floor(pos);
        double frac = pos - i0;
        double a = i0 >= 0 && i0 < s.length ? s[i0] : 0.0d;
        double b = i0 + 1 >= 0 && i0 + 1 < s.length ? s[i0 + 1] : 0.0d;
        return a * (1.0d - frac) + b * frac;
    }

    private static double energy(short[] s) {
        double sum = 0.0d;
        for (short v : s) {
            sum += (double) v * v;
        }
        return sum;
    }

    private static short clamp(double v) {
        if (v > Short.MAX_VALUE) {
            return Short.MAX_VALUE;
        }
        if (v < Short.MIN_VALUE) {
            return Short.MIN_VALUE;
        }
        return (short) Math.round(v);
    }
}
