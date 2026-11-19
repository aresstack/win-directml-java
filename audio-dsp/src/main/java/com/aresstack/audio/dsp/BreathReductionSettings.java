package com.aresstack.audio.dsp;

/** Validated, defensive settings for {@link BreathReductionProcessor}; out-of-range input is clamped. */
public final class BreathReductionSettings {

    private final double sensitivity;
    private final double maxAttenuationDb;
    private final boolean speechProtection;
    private final double attackMs;
    private final double releaseMs;

    public BreathReductionSettings(double sensitivity, double maxAttenuationDb, boolean speechProtection,
                                   double attackMs, double releaseMs) {
        this.sensitivity = clamp(sensitivity, 0.0d, 1.0d, 0.5d);
        this.maxAttenuationDb = clamp(maxAttenuationDb, 0.0d, 80.0d, 12.0d);
        this.speechProtection = speechProtection;
        this.attackMs = clamp(attackMs, 0.0d, 500.0d, 5.0d);
        this.releaseMs = clamp(releaseMs, 1.0d, 5000.0d, 120.0d);
    }

    public double getSensitivity() {
        return sensitivity;
    }

    public double getMaxAttenuationDb() {
        return maxAttenuationDb;
    }

    public boolean isSpeechProtection() {
        return speechProtection;
    }

    public double getAttackMs() {
        return attackMs;
    }

    public double getReleaseMs() {
        return releaseMs;
    }

    private static double clamp(double value, double min, double max, double fallback) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return fallback;
        }
        return value < min ? min : (value > max ? max : value);
    }
}
