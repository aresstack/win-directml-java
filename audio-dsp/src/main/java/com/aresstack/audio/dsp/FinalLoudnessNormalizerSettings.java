package com.aresstack.audio.dsp;

/** Immutable configuration for the {@link FinalLoudnessNormalizer}. Values are clamped to sane ranges. */
public final class FinalLoudnessNormalizerSettings {

    /** How the target level is measured. LUFS/integrated loudness is intentionally not offered yet. */
    public enum Mode {
        TARGET_RMS,
        PEAK
    }

    private final Mode mode;
    private final double targetLevelDb;
    private final double maxTotalGainDb;
    private final double maxTotalAttenuationDb;
    private final double peakCeilingDb;
    private final boolean clippingProtection;
    private final boolean allowAmplification;
    private final boolean allowAttenuation;

    public FinalLoudnessNormalizerSettings(Mode mode, double targetLevelDb, double maxTotalGainDb,
                                           double maxTotalAttenuationDb, double peakCeilingDb,
                                           boolean clippingProtection, boolean allowAmplification,
                                           boolean allowAttenuation) {
        this.mode = mode == null ? Mode.TARGET_RMS : mode;
        this.targetLevelDb = clamp(targetLevelDb, -60.0d, 0.0d);
        this.maxTotalGainDb = clamp(maxTotalGainDb, 0.0d, 60.0d);
        this.maxTotalAttenuationDb = clamp(maxTotalAttenuationDb, 0.0d, 60.0d);
        this.peakCeilingDb = clamp(peakCeilingDb, -30.0d, 0.0d);
        this.clippingProtection = clippingProtection;
        this.allowAmplification = allowAmplification;
        this.allowAttenuation = allowAttenuation;
    }

    public Mode getMode() {
        return mode;
    }

    public double getTargetLevelDb() {
        return targetLevelDb;
    }

    public double getMaxTotalGainDb() {
        return maxTotalGainDb;
    }

    public double getMaxTotalAttenuationDb() {
        return maxTotalAttenuationDb;
    }

    public double getPeakCeilingDb() {
        return peakCeilingDb;
    }

    public boolean isClippingProtection() {
        return clippingProtection;
    }

    public boolean isAllowAmplification() {
        return allowAmplification;
    }

    public boolean isAllowAttenuation() {
        return allowAttenuation;
    }

    private static double clamp(double value, double min, double max) {
        if (Double.isNaN(value)) {
            return min;
        }
        return value < min ? min : (value > max ? max : value);
    }
}
