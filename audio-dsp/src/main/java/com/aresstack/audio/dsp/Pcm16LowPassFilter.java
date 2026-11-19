package com.aresstack.audio.dsp;

/**
 * A linear-phase windowed-sinc FIR low-pass filter for 16-bit mono PCM. Pure and stateless. Used as the
 * anti-aliasing stage before downsampling: content above the destination Nyquist is removed first, so it
 * cannot fold back (alias) into the audible band.
 */
public final class Pcm16LowPassFilter {

    /** Odd tap count → a symmetric (linear-phase) kernel with a well-defined centre sample. */
    private static final int TAPS = 65;

    private Pcm16LowPassFilter() {
    }

    /**
     * @param mono       mono 16-bit samples
     * @param sampleRate the sample rate of {@code mono} (Hz)
     * @param cutoffHz   the low-pass cutoff (Hz); values ≥ Nyquist are a no-op (nothing to filter)
     * @return a new low-pass-filtered buffer of the same length
     */
    public static short[] filter(short[] mono, int sampleRate, double cutoffHz) {
        if (sampleRate <= 0) {
            throw new IllegalArgumentException("sampleRate must be positive");
        }
        int length = mono.length;
        double nyquist = sampleRate / 2.0d;
        if (length == 0 || cutoffHz <= 0.0d || cutoffHz >= nyquist) {
            short[] copy = new short[length];
            System.arraycopy(mono, 0, copy, 0, length);
            return copy;
        }

        double[] kernel = buildKernel(cutoffHz / sampleRate);
        int half = TAPS / 2;
        short[] output = new short[length];
        for (int i = 0; i < length; i++) {
            double accumulator = 0.0d;
            for (int k = 0; k < TAPS; k++) {
                int sampleIndex = i + k - half;
                if (sampleIndex >= 0 && sampleIndex < length) {
                    accumulator += kernel[k] * mono[sampleIndex];
                }
            }
            output[i] = clampToShort(Math.round(accumulator));
        }
        return output;
    }

    /** @param normalizedCutoff cutoff as a fraction of the sample rate (0..0.5). */
    private static double[] buildKernel(double normalizedCutoff) {
        double[] kernel = new double[TAPS];
        int half = TAPS / 2;
        double sum = 0.0d;
        for (int k = 0; k < TAPS; k++) {
            int n = k - half;
            double sinc = n == 0 ? 2.0d * normalizedCutoff
                    : Math.sin(2.0d * Math.PI * normalizedCutoff * n) / (Math.PI * n);
            // Hamming window keeps the stop-band attenuation high without excessive ripple.
            double window = 0.54d - 0.46d * Math.cos(2.0d * Math.PI * k / (TAPS - 1));
            kernel[k] = sinc * window;
            sum += kernel[k];
        }
        // Normalise to unity DC gain so the pass-band level is preserved.
        for (int k = 0; k < TAPS; k++) {
            kernel[k] /= sum;
        }
        return kernel;
    }

    private static short clampToShort(long value) {
        if (value > Short.MAX_VALUE) {
            return Short.MAX_VALUE;
        }
        if (value < Short.MIN_VALUE) {
            return Short.MIN_VALUE;
        }
        return (short) value;
    }
}
