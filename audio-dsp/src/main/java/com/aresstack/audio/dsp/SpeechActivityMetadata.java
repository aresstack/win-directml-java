package com.aresstack.audio.dsp;

/**
 * Immutable per-frame result of the voice-activity detector: a smoothed speech probability, the stabilized
 * speech-active state, and the frame's measured level and estimated noise floor (both in dBFS). All values
 * are guaranteed finite. This is analysis metadata only — it never changes the audio.
 */
public final class SpeechActivityMetadata {

    private final double speechProbability;
    private final boolean speechActive;
    private final double estimatedNoiseLevelDb;
    private final double measuredLevelDb;

    public SpeechActivityMetadata(double speechProbability, boolean speechActive,
                                  double estimatedNoiseLevelDb, double measuredLevelDb) {
        this.speechProbability = clampProbability(speechProbability);
        this.speechActive = speechActive;
        this.estimatedNoiseLevelDb = finite(estimatedNoiseLevelDb);
        this.measuredLevelDb = finite(measuredLevelDb);
    }

    public double getSpeechProbability() {
        return speechProbability;
    }

    public boolean isSpeechActive() {
        return speechActive;
    }

    public double getEstimatedNoiseLevelDb() {
        return estimatedNoiseLevelDb;
    }

    public double getMeasuredLevelDb() {
        return measuredLevelDb;
    }

    private static double clampProbability(double value) {
        if (Double.isNaN(value)) {
            return 0.0d;
        }
        if (value < 0.0d) {
            return 0.0d;
        }
        if (value > 1.0d) {
            return 1.0d;
        }
        return value;
    }

    private static double finite(double value) {
        if (Double.isNaN(value)) {
            return -120.0d;
        }
        if (value == Double.NEGATIVE_INFINITY) {
            return -120.0d;
        }
        if (value == Double.POSITIVE_INFINITY) {
            return 0.0d;
        }
        return value;
    }
}
