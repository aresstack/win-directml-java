package com.aresstack.audio.dsp;

/**
 * Modifies one STFT frame's spectrum in place. Multiply real and imaginary parts by the same per-bin gain
 * to change magnitude while preserving phase. Implementations may keep cross-frame state (a fresh instance
 * is created per run). {@code frameStartSample} is the mono sample index of the frame start, so a modifier
 * can line up with per-frame metadata such as the speech-activity track.
 */
public interface SpectralModifier {

    void modify(double[] real, double[] imag, int sampleRateHz, int frameStartSample);
}
