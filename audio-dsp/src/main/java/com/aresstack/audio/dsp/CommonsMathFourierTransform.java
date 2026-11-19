package com.aresstack.audio.dsp;

import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;

/**
 * {@link FourierTransform} backed by Apache Commons Math (Java 8 compatible, no runtime dependencies). Uses
 * the standard normalization so an inverse after a forward reproduces the input; requires power-of-two length.
 */
public final class CommonsMathFourierTransform implements FourierTransform {

    public void forward(double[] real, double[] imag) {
        transform(real, imag, TransformType.FORWARD);
    }

    public void inverse(double[] real, double[] imag) {
        transform(real, imag, TransformType.INVERSE);
    }

    private static void transform(double[] real, double[] imag, TransformType type) {
        if (real.length != imag.length) {
            throw new IllegalArgumentException("Real and imaginary parts must have the same length.");
        }
        if (Integer.bitCount(real.length) != 1) {
            throw new IllegalArgumentException("FFT length must be a power of two, got " + real.length + ".");
        }
        // transformInPlace mutates the supplied real/imag arrays in place.
        FastFourierTransformer.transformInPlace(new double[][]{real, imag}, DftNormalization.STANDARD, type);
    }
}
