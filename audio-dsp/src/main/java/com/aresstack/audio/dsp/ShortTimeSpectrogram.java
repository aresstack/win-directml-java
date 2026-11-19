package com.aresstack.audio.dsp;

/**
 * A full complex short-time spectrogram of a mono signal (all frames kept), with a matching weighted
 * overlap-add inverse. Unlike {@link ShortTimeFourierTransform}, which applies a per-frame modifier and
 * reconstructs in one pass, this exposes the whole time-frequency matrix so an algorithm (for example WPE
 * dereverberation) can process each frequency bin along the time axis before reconstruction. With no change
 * to the matrix the inverse reproduces the input (within rounding).
 */
public final class ShortTimeSpectrogram {

    private static final double EPS = 1.0e-9d;

    private final int frameSize;
    private final int hop;
    private final int signalLength;
    private final double[] window;
    private final FourierTransform fft;
    private final double[][] real;
    private final double[][] imag;

    private ShortTimeSpectrogram(int frameSize, int hop, int signalLength, double[] window,
                                 FourierTransform fft, double[][] real, double[][] imag) {
        this.frameSize = frameSize;
        this.hop = hop;
        this.signalLength = signalLength;
        this.window = window;
        this.fft = fft;
        this.real = real;
        this.imag = imag;
    }

    /** Forward-transform a mono signal into its complex spectrogram. */
    public static ShortTimeSpectrogram forward(double[] mono, int frameSize, int hop) {
        if (Integer.bitCount(frameSize) != 1) {
            throw new IllegalArgumentException("Frame size must be a power of two.");
        }
        if (hop <= 0 || hop > frameSize) {
            throw new IllegalArgumentException("Hop must be in (0, frameSize].");
        }
        double[] window = WindowFunctions.hann(frameSize);
        FourierTransform fft = new CommonsMathFourierTransform();
        int length = mono.length;
        int padded = length + 2 * frameSize;
        double[] input = new double[padded];
        System.arraycopy(mono, 0, input, frameSize, length);
        int frameCount = 0;
        for (int start = 0; start + frameSize <= padded; start += hop) {
            frameCount++;
        }
        double[][] real = new double[frameCount][frameSize];
        double[][] imag = new double[frameCount][frameSize];
        int f = 0;
        for (int start = 0; start + frameSize <= padded; start += hop, f++) {
            double[] re = real[f];
            double[] im = imag[f];
            for (int i = 0; i < frameSize; i++) {
                re[i] = input[start + i] * window[i];
                im[i] = 0.0d;
            }
            fft.forward(re, im);
        }
        return new ShortTimeSpectrogram(frameSize, hop, length, window, fft, real, imag);
    }

    public int getFrameCount() {
        return real.length;
    }

    public int getFrameSize() {
        return frameSize;
    }

    public double[] realFrame(int frame) {
        return real[frame];
    }

    public double[] imagFrame(int frame) {
        return imag[frame];
    }

    /** Reconstruct the mono signal from the (possibly modified) spectrogram via weighted overlap-add. */
    public double[] inverse() {
        int padded = signalLength + 2 * frameSize;
        double[] output = new double[padded];
        double[] norm = new double[padded];
        double[] re = new double[frameSize];
        double[] im = new double[frameSize];
        int start = 0;
        for (int f = 0; f < real.length; f++, start += hop) {
            System.arraycopy(real[f], 0, re, 0, frameSize);
            System.arraycopy(imag[f], 0, im, 0, frameSize);
            fft.inverse(re, im);
            for (int i = 0; i < frameSize; i++) {
                output[start + i] += re[i] * window[i];
                norm[start + i] += window[i] * window[i];
            }
        }
        double[] result = new double[signalLength];
        for (int j = 0; j < signalLength; j++) {
            double n = norm[frameSize + j];
            result[j] = n > EPS ? output[frameSize + j] / n : 0.0d;
        }
        return result;
    }
}
