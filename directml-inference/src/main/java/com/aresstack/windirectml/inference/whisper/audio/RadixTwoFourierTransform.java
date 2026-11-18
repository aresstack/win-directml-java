package com.aresstack.windirectml.inference.whisper.audio;

/**
 * The FFT of the core: iterative radix 2, with the twiddle factors and the bit reversal computed
 * once in the constructor (architecture review 4b.2: as fast as Commons Math at the lengths of
 * the convolution, without allocation, and more exact). Forward uses {@code e^{-2πikn/N}}; the
 * inverse is normalized by {@code 1/N}.
 *
 * <p>An instance is bound to its length and keeps no state between calls other than its
 * read-only tables, so one instance may serve several channels on one thread.
 */
public final class RadixTwoFourierTransform implements FourierTransform {
    private final int length;
    private final double[] cos;
    private final double[] sin;
    private final int[] reversed;

    /** @param length a power of two, at least 2 */
    public RadixTwoFourierTransform(int length) {
        if (length < 2 || Integer.bitCount(length) != 1) {
            throw new IllegalArgumentException("An FFT needs a power of two of at least 2, not " + length + ".");
        }
        this.length = length;
        cos = new double[length / 2];
        sin = new double[length / 2];
        for (int i = 0; i < length / 2; i++) {
            double angle = 2 * Math.PI * i / length;
            cos[i] = Math.cos(angle);
            sin[i] = Math.sin(angle);
        }
        reversed = new int[length];
        int bits = Integer.numberOfTrailingZeros(length);
        for (int i = 0; i < length; i++) {
            reversed[i] = Integer.reverse(i) >>> (32 - bits);
        }
    }

    public int length() {
        return length;
    }

    @Override
    public void forward(double[] real, double[] imag) {
        transform(real, imag, -1);
    }

    @Override
    public void inverse(double[] real, double[] imag) {
        transform(real, imag, 1);
        double scale = 1.0 / length;
        for (int i = 0; i < length; i++) {
            real[i] *= scale;
            imag[i] *= scale;
        }
    }

    private void transform(double[] real, double[] imag, double sign) {
        if (real.length != length || imag.length != length) {
            // A wrong length is a programming error found in the first test, never on stage.
            throw new IllegalArgumentException("This FFT is bound to " + length + " points.");
        }
        int n = length;
        for (int i = 0; i < n; i++) {
            int j = reversed[i];
            if (j > i) {
                double t = real[i];
                real[i] = real[j];
                real[j] = t;
                t = imag[i];
                imag[i] = imag[j];
                imag[j] = t;
            }
        }
        for (int size = 2; size <= n; size <<= 1) {
            int half = size >>> 1;
            int step = n / size;
            for (int start = 0; start < n; start += size) {
                for (int k = 0, t = 0; k < half; k++, t += step) {
                    double wr = cos[t];
                    double wi = sign * sin[t];
                    int a = start + k;
                    int b = a + half;
                    double xr = real[b] * wr - imag[b] * wi;
                    double xi = real[b] * wi + imag[b] * wr;
                    real[b] = real[a] - xr;
                    imag[b] = imag[a] - xi;
                    real[a] += xr;
                    imag[a] += xi;
                }
            }
        }
    }
}
