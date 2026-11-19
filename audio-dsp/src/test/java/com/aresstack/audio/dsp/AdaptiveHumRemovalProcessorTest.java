package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

/** Adaptive hum removal attenuates the mains fundamental (even when it drifts) while keeping other tones. */
public class AdaptiveHumRemovalProcessorTest {

    private static final int RATE = 16000;
    private static final PcmAudioFormat MONO = new PcmAudioFormat(RATE, 1, 16);

    @Test
    public void attenuatesASteady50HzHum() {
        short[] input = mix(1000.0d, 6000, 50.0d, 4000, RATE);
        short[] out = process(new AdaptiveHumRemovalSettings(50.0d, 3.0d, 0.2d, 1, 30.0d, false), input);
        assertTrue("50 Hz hum reduced", powerAt(out, 50.0d) < 0.3d * powerAt(input, 50.0d));
        assertTrue("1 kHz content preserved", powerAt(out, 1000.0d) > 0.7d * powerAt(input, 1000.0d));
    }

    @Test
    public void tracksAndAttenuatesADriftedFundamental() {
        short[] input = mix(1000.0d, 6000, 53.0d, 4000, RATE);
        short[] out = process(new AdaptiveHumRemovalSettings(50.0d, 4.0d, 0.5d, 1, 30.0d, false), input);
        assertTrue("drifted 53 Hz hum is tracked and reduced",
                powerAt(out, 53.0d) < 0.4d * powerAt(input, 53.0d));
    }

    @Test
    public void silenceStaysStable() {
        short[] out = process(new AdaptiveHumRemovalSettings(50.0d, 3.0d, 0.2d, 3, 24.0d, false), new short[RATE]);
        for (short sample : out) {
            assertTrue(sample == 0);
        }
    }

    private static short[] process(AdaptiveHumRemovalSettings s, short[] input) {
        short[] copy = input.clone();
        new AdaptiveHumRemovalProcessor(s).process(copy, copy.length, MONO, null);
        return copy;
    }

    private static short[] mix(double f1, int a1, double f2, int a2, int n) {
        short[] out = new short[n];
        for (int i = 0; i < n; i++) {
            double v = a1 * Math.sin(2.0d * Math.PI * f1 * i / RATE) + a2 * Math.sin(2.0d * Math.PI * f2 * i / RATE);
            out[i] = (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, Math.round(v)));
        }
        return out;
    }

    private static double powerAt(short[] samples, double freq) {
        double[] mono = new double[samples.length];
        for (int i = 0; i < samples.length; i++) {
            mono[i] = samples[i];
        }
        // Measure over the second half (after the tracker has converged).
        int from = samples.length / 2;
        return Goertzel.power(mono, from, samples.length - from, RATE, freq);
    }
}
