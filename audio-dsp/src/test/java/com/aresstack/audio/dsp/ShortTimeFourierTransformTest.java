package com.aresstack.audio.dsp;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

/** The STFT reconstructs the signal with an identity modifier (weighted overlap-add is exact). */
public class ShortTimeFourierTransformTest {

    private static final int RATE = 16000;

    @Test
    public void identityModifierReconstructsTheSignal() {
        ShortTimeFourierTransform stft =
                new ShortTimeFourierTransform(1024, 512, new CommonsMathFourierTransform());
        double[] mono = signal(20000);
        double[] out = stft.process(mono, RATE, new SpectralModifier() {
            public void modify(double[] real, double[] imag, int sampleRateHz, int frameStartSample) {
                // no-op
            }
        });
        double maxDiff = 0.0d;
        for (int i = 0; i < mono.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(mono[i] - out[i]));
        }
        assertTrue("WOLA reconstruction should be near-exact, max diff " + maxDiff, maxDiff < 1.0e-3d);
    }

    @Test
    public void aBinGainAttenuatesThatFrequency() {
        ShortTimeFourierTransform stft =
                new ShortTimeFourierTransform(1024, 512, new CommonsMathFourierTransform());
        double[] mono = tone(1000.0d, 20000, 8000);
        double[] out = stft.process(mono, RATE, new SpectralModifier() {
            public void modify(double[] real, double[] imag, int sampleRateHz, int frameStartSample) {
                int bin = Math.round(1000.0f / (RATE / 1024.0f));
                for (int k = bin - 2; k <= bin + 2; k++) {
                    if (k >= 1 && k <= 512) {
                        Spectra.applyGain(real, imag, k, 0.1d);
                    }
                }
            }
        });
        assertTrue("attenuating the 1 kHz bins lowers the tone", rms(out) < 0.4d * rms(mono));
    }

    private static double[] signal(int n) {
        double[] out = new double[n];
        int state = 98765;
        for (int i = 0; i < n; i++) {
            state = state * 1103515245 + 12345;
            out[i] = (state >> 16) % 8000
                    + 3000.0d * Math.sin(2.0d * Math.PI * 440.0d * i / RATE);
        }
        return out;
    }

    private static double[] tone(double freq, int n, int amp) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = amp * Math.sin(2.0d * Math.PI * freq * i / RATE);
        }
        return out;
    }

    private static double rms(double[] samples) {
        double sum = 0.0d;
        int from = samples.length / 2;
        for (int i = from; i < samples.length; i++) {
            sum += samples[i] * samples[i];
        }
        return Math.sqrt(sum / (samples.length - from));
    }
}
