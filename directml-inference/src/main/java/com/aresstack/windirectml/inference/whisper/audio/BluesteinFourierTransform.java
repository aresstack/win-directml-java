package com.aresstack.windirectml.inference.whisper.audio;

import java.util.Arrays;

/**
 * Compute arbitrary-length Fourier transforms using Bluestein convolution and the
 * radix-two implementation copied from the shared audio core.
 * Keep the underlying transform independent of Whisper's feature extraction.
 */
public final class BluesteinFourierTransform implements FourierTransform {
    private final int size;
    private final int convolutionSize;
    private final RadixTwoFourierTransform fft;
    private final double[] chirpCos;
    private final double[] chirpSin;
    private final double[] kernelReal;
    private final double[] kernelImag;

    public BluesteinFourierTransform(int size) {
        if (size < 2 || size > (1 << 18)) {
            throw new IllegalArgumentException("Unsupported transform size: " + size);
        }
        this.size = size;
        int convolutionLength = 1;
        while (convolutionLength < 2 * size - 1) {
            convolutionLength <<= 1;
        }
        this.convolutionSize = convolutionLength;
        this.fft = new RadixTwoFourierTransform(convolutionSize);
        this.chirpCos = new double[size];
        this.chirpSin = new double[size];
        this.kernelReal = new double[convolutionSize];
        this.kernelImag = new double[convolutionSize];
        for (int i = 0; i < size; i++) {
            double phase = Math.PI * ((long) i * i % (2L * size)) / size;
            chirpCos[i] = Math.cos(phase);
            chirpSin[i] = Math.sin(phase);
            kernelReal[i] = chirpCos[i];
            kernelImag[i] = chirpSin[i];
            if (i > 0) {
                kernelReal[convolutionSize - i] = chirpCos[i];
                kernelImag[convolutionSize - i] = chirpSin[i];
            }
        }
        fft.forward(kernelReal, kernelImag);
    }

    @Override
    public void forward(double[] real, double[] imag) {
        validate(real, imag);
        double[] convolutionReal = new double[convolutionSize];
        double[] convolutionImag = new double[convolutionSize];
        for (int i = 0; i < size; i++) {
            double re = real[i];
            double im = imag[i];
            convolutionReal[i] = re * chirpCos[i] + im * chirpSin[i];
            convolutionImag[i] = im * chirpCos[i] - re * chirpSin[i];
        }
        fft.forward(convolutionReal, convolutionImag);
        for (int i = 0; i < convolutionSize; i++) {
            double re = convolutionReal[i];
            double im = convolutionImag[i];
            convolutionReal[i] = re * kernelReal[i] - im * kernelImag[i];
            convolutionImag[i] = re * kernelImag[i] + im * kernelReal[i];
        }
        fft.inverse(convolutionReal, convolutionImag);
        for (int i = 0; i < size; i++) {
            double re = convolutionReal[i];
            double im = convolutionImag[i];
            real[i] = re * chirpCos[i] + im * chirpSin[i];
            imag[i] = im * chirpCos[i] - re * chirpSin[i];
        }
    }

    @Override
    public void inverse(double[] real, double[] imag) {
        validate(real, imag);
        for (int i = 0; i < size; i++) {
            imag[i] = -imag[i];
        }
        forward(real, imag);
        for (int i = 0; i < size; i++) {
            real[i] /= size;
            imag[i] = -imag[i] / size;
        }
    }

    private void validate(double[] real, double[] imag) {
        if (real == null || imag == null || real.length != size || imag.length != size) {
            throw new IllegalArgumentException("Expected two complex arrays of length " + size);
        }
    }
}
