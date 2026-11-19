package com.aresstack.audio.dsp;

/** Validated, defensive settings for {@link DeEsserProcessor}; out-of-range input is clamped. */
public final class DeEsserSettings {

    private final double targetFrequencyHz;
    private final double bandwidthHz;
    private final double thresholdDb;
    private final double reductionDb;
    private final double attackMs;
    private final double releaseMs;

    public DeEsserSettings(double targetFrequencyHz, double bandwidthHz, double thresholdDb,
                           double reductionDb, double attackMs, double releaseMs) {
        this.targetFrequencyHz = clamp(targetFrequencyHz, 1000.0d, 20000.0d, 6500.0d);
        this.bandwidthHz = clamp(bandwidthHz, 100.0d, 12000.0d, 2500.0d);
        this.thresholdDb = clamp(thresholdDb, -80.0d, 0.0d, -30.0d);
        this.reductionDb = clamp(reductionDb, 0.0d, 40.0d, 8.0d);
        this.attackMs = clamp(attackMs, 0.0d, 200.0d, 2.0d);
        this.releaseMs = clamp(releaseMs, 1.0d, 2000.0d, 60.0d);
    }

    public double getTargetFrequencyHz() {
        return targetFrequencyHz;
    }

    public double getBandwidthHz() {
        return bandwidthHz;
    }

    public double getThresholdDb() {
        return thresholdDb;
    }

    public double getReductionDb() {
        return reductionDb;
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
