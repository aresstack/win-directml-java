package com.aresstack.audio.dsp;

/**
 * Short-time Fourier transform with weighted overlap-add: split a mono signal into overlapping,
 * Hann-windowed frames, transform each, let a {@link SpectralModifier} adjust the spectrum, invert and
 * overlap-add with per-sample window-energy normalization. With an identity modifier it reconstructs the
 * input (within rounding); the signal is padded front and back so edge samples are fully covered.
 */
public final class ShortTimeFourierTransform {

    private static final double EPS = 1.0e-9d;

    private final int frameSize;
    private final int hop;
    private final FourierTransform fft;
    private final double[] window;

    public ShortTimeFourierTransform(int frameSize, int hop, FourierTransform fft) {
        if (Integer.bitCount(frameSize) != 1) {
            throw new IllegalArgumentException("Frame size must be a power of two.");
        }
        if (hop <= 0 || hop > frameSize) {
            throw new IllegalArgumentException("Hop must be in (0, frameSize].");
        }
        if (fft == null) {
            throw new IllegalArgumentException("Fourier transform must not be null.");
        }
        this.frameSize = frameSize;
        this.hop = hop;
        this.fft = fft;
        this.window = WindowFunctions.hann(frameSize);
    }

    /** @return a new mono signal of the same length, with {@code modifier} applied per frame. */
    public double[] process(double[] mono, int sampleRateHz, SpectralModifier modifier) {
        int length = mono.length;
        int padded = length + 2 * frameSize;
        double[] input = new double[padded];
        System.arraycopy(mono, 0, input, frameSize, length);
        double[] output = new double[padded];
        double[] norm = new double[padded];

        double[] real = new double[frameSize];
        double[] imag = new double[frameSize];
        for (int start = 0; start + frameSize <= padded; start += hop) {
            for (int i = 0; i < frameSize; i++) {
                real[i] = input[start + i] * window[i];
                imag[i] = 0.0d;
            }
            fft.forward(real, imag);
            modifier.modify(real, imag, sampleRateHz, start - frameSize);
            fft.inverse(real, imag);
            for (int i = 0; i < frameSize; i++) {
                output[start + i] += real[i] * window[i];
                norm[start + i] += window[i] * window[i];
            }
        }

        double[] result = new double[length];
        for (int j = 0; j < length; j++) {
            double n = norm[frameSize + j];
            result[j] = n > EPS ? output[frameSize + j] / n : 0.0d;
        }
        return result;
    }
}
