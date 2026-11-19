package com.aresstack.audio.dsp;

/**
 * Estimated room/reverberation characteristics of a recording: a reverberation time (an RT60-style estimate
 * in seconds), a normalized reverberation strength in [0, 1] and a confidence in [0, 1]. Produced by the
 * Room/Reverb Analyzer and consumed by a later Dereverberation block through the processing context.
 *
 * <p>Transient per-run metadata; never persisted as a profile parameter.</p>
 */
public final class RoomProfile {

    private final int sampleRateHz;
    private final double reverbTimeSeconds;
    private final double reverbStrength;
    private final double confidence;

    public RoomProfile(int sampleRateHz, double reverbTimeSeconds, double reverbStrength, double confidence) {
        this.sampleRateHz = sampleRateHz;
        this.reverbTimeSeconds = Math.max(0.0d, finite(reverbTimeSeconds));
        this.reverbStrength = clamp01(reverbStrength);
        this.confidence = clamp01(confidence);
    }

    public int getSampleRateHz() {
        return sampleRateHz;
    }

    /** @return the estimated reverberation time (RT60-style), in seconds. */
    public double getReverbTimeSeconds() {
        return reverbTimeSeconds;
    }

    /** @return a normalized reverberation strength in [0, 1] (0 = dry, 1 = very reverberant). */
    public double getReverbStrength() {
        return reverbStrength;
    }

    public double getConfidence() {
        return confidence;
    }

    private static double clamp01(double value) {
        if (Double.isNaN(value) || value < 0.0d) {
            return 0.0d;
        }
        return value > 1.0d ? 1.0d : value;
    }

    private static double finite(double value) {
        return Double.isNaN(value) || Double.isInfinite(value) ? 0.0d : value;
    }
}
