package com.aresstack.audio.dsp;

/**
 * A minimal in-place FFT port: forward and inverse transforms of a power-of-two complex buffer given as
 * separate real/imaginary arrays. Abstracted so the spectral blocks depend on this interface, not directly
 * on the FFT library, and so an alternative implementation can be swapped in without touching the blocks.
 */
public interface FourierTransform {

    /** Forward transform in place; {@code real} and {@code imag} must have the same power-of-two length. */
    void forward(double[] real, double[] imag);

    /** Inverse transform in place (normalized so forward-then-inverse returns the original signal). */
    void inverse(double[] real, double[] imag);
}
