package com.aresstack.audio.dsp;

/** Small helpers shared by the spectral modifiers: bin/gain math on a real-signal half spectrum. */
final class Spectra {

    static final double FULL_SCALE = 32768.0d;
    static final double EPS = 1.0e-9d;

    private Spectra() {
    }

    /** Apply gain {@code g} to bin {@code k} and its conjugate mirror, preserving phase. */
    static void applyGain(double[] real, double[] imag, int k, double g) {
        int n = real.length;
        real[k] *= g;
        imag[k] *= g;
        int mirror = n - k;
        if (mirror > k && mirror < n) {
            real[mirror] *= g;
            imag[mirror] *= g;
        }
    }

    static double magnitude(double[] real, double[] imag, int k) {
        return Math.sqrt(real[k] * real[k] + imag[k] * imag[k]);
    }

    /** One-pole smoothing coefficient for a time constant, evaluated per hop. */
    static double coefficient(double timeMs, double hopMs) {
        double tau = Math.max(0.001d, timeMs);
        return 1.0d - Math.exp(-hopMs / tau);
    }

    static double clamp01(double value) {
        if (Double.isNaN(value) || value < 0.0d) {
            return 0.0d;
        }
        return value > 1.0d ? 1.0d : value;
    }

    static int binOf(double frequencyHz, double binHz, int maxBin) {
        int bin = (int) Math.round(frequencyHz / binHz);
        if (bin < 1) {
            return 1;
        }
        return bin > maxBin ? maxBin : bin;
    }
}
