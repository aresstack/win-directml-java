package com.aresstack.windirectml.inference.whisper.audio;

/**
 * A minimal in-place FFT port: forward and inverse transforms of a power-of-two complex buffer
 * given as separate real and imaginary arrays.
 *
 * <p>Taken over unchanged from AskAI's {@code audio-dsp} module
 * ({@code com.aresstack.audio.dsp.FourierTransform}, MIT, AskAI {@code develop} {@code 5b788e2});
 * only its package differs (architecture review 4b.2, decision 1). The core adds to the contract:
 * an instance is bound to one length, so its tables exist before the first call, and
 * {@link #forward} and {@link #inverse} neither allocate nor lock; they may be called on the audio
 * thread.
 */
public interface FourierTransform {

    /** Forward transform in place; {@code real} and {@code imag} must have the same power-of-two length. */
    void forward(double[] real, double[] imag);

    /** Inverse transform in place (normalized so forward-then-inverse returns the original signal). */
    void inverse(double[] real, double[] imag);
}
