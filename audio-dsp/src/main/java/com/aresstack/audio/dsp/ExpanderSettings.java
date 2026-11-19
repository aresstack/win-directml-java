package com.aresstack.audio.dsp;

/**
 * Validated, defensive settings for {@link ExpanderProcessor}. Every value is clamped to a sane range and
 * non-finite input falls back to a default, so the expander can never be driven into NaN/Infinity, a
 * discontinuous gain jump or a wrap-around, even if a profile carries out-of-range parameters.
 */
public final class ExpanderSettings {

    private final double thresholdDb;
    private final double ratio;
    private final double kneeDb;
    private final double attackMs;
    private final double releaseMs;
    private final double holdMs;
    private final double maxAttenuationDb;
    private final double detectorWindowMs;
    private final boolean speechProtection;
    private final double minSpeechProbability;

    public ExpanderSettings(double thresholdDb, double ratio, double kneeDb, double attackMs, double releaseMs,
                            double holdMs, double maxAttenuationDb, double detectorWindowMs,
                            boolean speechProtection, double minSpeechProbability) {
        this.thresholdDb = clamp(thresholdDb, -120.0d, 0.0d, -45.0d);
        this.ratio = clamp(ratio, 1.0d, 20.0d, 2.0d);
        this.kneeDb = clamp(kneeDb, 0.0d, 24.0d, 6.0d);
        this.attackMs = clamp(attackMs, 0.0d, 500.0d, 10.0d);
        this.releaseMs = clamp(releaseMs, 10.0d, 5000.0d, 200.0d);
        this.holdMs = clamp(holdMs, 0.0d, 2000.0d, 50.0d);
        this.maxAttenuationDb = clamp(maxAttenuationDb, 0.0d, 80.0d, 18.0d);
        this.detectorWindowMs = clamp(detectorWindowMs, 5.0d, 100.0d, 20.0d);
        this.speechProtection = speechProtection;
        this.minSpeechProbability = clamp(minSpeechProbability, 0.0d, 1.0d, 0.5d);
    }

    public double getThresholdDb() {
        return thresholdDb;
    }

    public double getRatio() {
        return ratio;
    }

    public double getKneeDb() {
        return kneeDb;
    }

    public double getAttackMs() {
        return attackMs;
    }

    public double getReleaseMs() {
        return releaseMs;
    }

    public double getHoldMs() {
        return holdMs;
    }

    public double getMaxAttenuationDb() {
        return maxAttenuationDb;
    }

    public double getDetectorWindowMs() {
        return detectorWindowMs;
    }

    public boolean isSpeechProtection() {
        return speechProtection;
    }

    public double getMinSpeechProbability() {
        return minSpeechProbability;
    }

    private static double clamp(double value, double min, double max, double fallback) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return fallback;
        }
        if (value < min) {
            return min;
        }
        if (value > max) {
            return max;
        }
        return value;
    }
}
