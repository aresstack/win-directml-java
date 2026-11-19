package com.aresstack.audio.dsp;

/**
 * Single-bin Goertzel power estimate — the minimal spectral probe needed to track a narrow tone (mains
 * hum) without a full FFT. Given a window of samples it returns the energy at one target frequency, so the
 * adaptive hum remover can scan a small set of candidate frequencies and pick the strongest.
 */
public final class Goertzel {

    private Goertzel() {
    }

    /** @return the (length-normalized) power at {@code frequencyHz} over {@code count} samples from {@code offset}. */
    public static double power(double[] samples, int offset, int count, int sampleRateHz, double frequencyHz) {
        if (count <= 0 || sampleRateHz <= 0) {
            return 0.0d;
        }
        double omega = 2.0d * Math.PI * frequencyHz / sampleRateHz;
        double coeff = 2.0d * Math.cos(omega);
        double s1 = 0.0d;
        double s2 = 0.0d;
        int end = Math.min(samples.length, offset + count);
        int n = 0;
        for (int i = offset; i < end; i++) {
            double s0 = samples[i] + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
            n++;
        }
        if (n == 0) {
            return 0.0d;
        }
        double power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
        return Math.max(0.0d, power) / n;
    }
}
