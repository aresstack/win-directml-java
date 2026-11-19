package com.aresstack.audio.dsp;

/** Validated, defensive settings for {@link PlosiveReductionProcessor}; out-of-range input is clamped. */
public final class PlosiveReductionSettings {

    private final double strength;
    private final double targetFrequencyHz;
    private final double attackMs;
    private final double releaseMs;

    public PlosiveReductionSettings(double strength, double targetFrequencyHz, double attackMs,
                                    double releaseMs) {
        this.strength = clamp(strength, 0.0d, 1.0d, 0.6d);
        this.targetFrequencyHz = clamp(targetFrequencyHz, 20.0d, 500.0d, 120.0d);
        this.attackMs = clamp(attackMs, 0.0d, 200.0d, 3.0d);
        this.releaseMs = clamp(releaseMs, 1.0d, 2000.0d, 80.0d);
    }

    public double getStrength() {
        return strength;
    }

    public double getTargetFrequencyHz() {
        return targetFrequencyHz;
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
