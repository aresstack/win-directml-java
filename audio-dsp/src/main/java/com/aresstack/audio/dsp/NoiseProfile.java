package com.aresstack.audio.dsp;

/**
 * A learned background-noise model: the average magnitude of a half spectrum (bins 0..fftSize/2) estimated
 * from speech pauses or an explicit learn recording, plus the format it was learned at and a confidence.
 *
 * <p>This is transient runtime metadata produced by a Noise Profiler and consumed by Adaptive Noise
 * Suppression through the {@link com.aresstack.audio.pipeline.AudioProcessingContext}. It carries no
 * adaptive runtime state and is never persisted as a profile parameter.</p>
 */
public final class NoiseProfile {

    private final int sampleRateHz;
    private final int fftSize;
    private final double[] magnitude;
    private final double confidence;

    public NoiseProfile(int sampleRateHz, int fftSize, double[] magnitude, double confidence) {
        if (sampleRateHz <= 0) {
            throw new IllegalArgumentException("Sample rate must be positive.");
        }
        if (fftSize <= 0 || Integer.bitCount(fftSize) != 1) {
            throw new IllegalArgumentException("FFT size must be a positive power of two.");
        }
        if (magnitude == null || magnitude.length != fftSize / 2 + 1) {
            throw new IllegalArgumentException("Magnitude must have fftSize/2+1 bins.");
        }
        this.sampleRateHz = sampleRateHz;
        this.fftSize = fftSize;
        this.magnitude = magnitude.clone();
        this.confidence = confidence;
    }

    public int getSampleRateHz() {
        return sampleRateHz;
    }

    public int getFftSize() {
        return fftSize;
    }

    /** @return the per-bin average noise magnitude, length fftSize/2+1 (a defensive copy). */
    public double[] getMagnitude() {
        return magnitude.clone();
    }

    /** @return the magnitude of bin {@code k} without copying the array. */
    public double magnitudeAt(int k) {
        return magnitude[k];
    }

    /** @return how reliable the estimate is in [0, 1] (fraction of frames that were treated as noise). */
    public double getConfidence() {
        return confidence;
    }
}
