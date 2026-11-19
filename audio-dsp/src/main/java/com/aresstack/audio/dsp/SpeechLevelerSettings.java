package com.aresstack.audio.dsp;

/** Immutable configuration for the {@link SpeechLevelerProcessor}. Values are clamped to sane ranges. */
public final class SpeechLevelerSettings {

    private final double targetSpeechLevelDb;
    private final double maxGainDb;
    private final double maxAttenuationDb;
    private final double attackMs;
    private final double releaseMs;
    private final double holdMs;
    private final double maxGainChangePerSecond;
    private final double minSpeechProbability;
    private final double silenceGainLimitDb;
    private final boolean noiseProtection;
    private final boolean clippingProtection;

    public SpeechLevelerSettings(double targetSpeechLevelDb, double maxGainDb, double maxAttenuationDb,
                                 double attackMs, double releaseMs, double holdMs,
                                 double maxGainChangePerSecond, double minSpeechProbability,
                                 double silenceGainLimitDb, boolean noiseProtection, boolean clippingProtection) {
        this.targetSpeechLevelDb = clamp(targetSpeechLevelDb, -60.0d, 0.0d);
        this.maxGainDb = clamp(maxGainDb, 0.0d, 48.0d);
        this.maxAttenuationDb = clamp(maxAttenuationDb, 0.0d, 48.0d);
        this.attackMs = Math.max(1.0d, attackMs);
        this.releaseMs = Math.max(1.0d, releaseMs);
        this.holdMs = Math.max(0.0d, holdMs);
        this.maxGainChangePerSecond = clamp(maxGainChangePerSecond, 0.1d, 60.0d);
        this.minSpeechProbability = clamp(minSpeechProbability, 0.0d, 1.0d);
        this.silenceGainLimitDb = clamp(silenceGainLimitDb, 0.0d, 24.0d);
        this.noiseProtection = noiseProtection;
        this.clippingProtection = clippingProtection;
    }

    public double getTargetSpeechLevelDb() {
        return targetSpeechLevelDb;
    }

    public double getMaxGainDb() {
        return maxGainDb;
    }

    public double getMaxAttenuationDb() {
        return maxAttenuationDb;
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

    public double getMaxGainChangePerSecond() {
        return maxGainChangePerSecond;
    }

    public double getMinSpeechProbability() {
        return minSpeechProbability;
    }

    public double getSilenceGainLimitDb() {
        return silenceGainLimitDb;
    }

    public boolean isNoiseProtection() {
        return noiseProtection;
    }

    public boolean isClippingProtection() {
        return clippingProtection;
    }

    private static double clamp(double value, double min, double max) {
        if (Double.isNaN(value)) {
            return min;
        }
        return value < min ? min : (value > max ? max : value);
    }
}
