package com.aresstack.audio.dsp;

/** Immutable configuration for {@link SpectralNoiseSuppressor}. All values are clamped to sane ranges. */
public final class NoiseSuppressionSettings {

    /** How the noise estimate is obtained. */
    public enum Mode {
        AUTOMATIC,
        LEARN_FROM_SILENCE,
        USE_FIXED_PROFILE
    }

    private final Mode mode;
    private final double maxAttenuationDb;
    private final double adaptationSpeed;
    private final double noiseFloorDb;
    private final boolean speechProtection;
    private final double minSpeechProbability;
    private final boolean adaptDuringSpeech;
    private final boolean freezeProfile;
    private final double artifactProtection;
    private final double attackMs;
    private final double releaseMs;

    public NoiseSuppressionSettings(Mode mode, double maxAttenuationDb, double adaptationSpeed,
                                    double noiseFloorDb, boolean speechProtection, double minSpeechProbability,
                                    boolean adaptDuringSpeech, boolean freezeProfile, double artifactProtection,
                                    double attackMs, double releaseMs) {
        this.mode = mode == null ? Mode.AUTOMATIC : mode;
        this.maxAttenuationDb = clamp(maxAttenuationDb, 0.0d, 80.0d);
        this.adaptationSpeed = clamp(adaptationSpeed, 0.0d, 1.0d);
        this.noiseFloorDb = clamp(noiseFloorDb, -120.0d, 0.0d);
        this.speechProtection = speechProtection;
        this.minSpeechProbability = clamp(minSpeechProbability, 0.0d, 1.0d);
        this.adaptDuringSpeech = adaptDuringSpeech;
        this.freezeProfile = freezeProfile;
        this.artifactProtection = clamp(artifactProtection, 0.0d, 1.0d);
        this.attackMs = Math.max(0.0d, attackMs);
        this.releaseMs = Math.max(0.0d, releaseMs);
    }

    public Mode getMode() {
        return mode;
    }

    public double getMaxAttenuationDb() {
        return maxAttenuationDb;
    }

    public double getAdaptationSpeed() {
        return adaptationSpeed;
    }

    public double getNoiseFloorDb() {
        return noiseFloorDb;
    }

    public boolean isSpeechProtection() {
        return speechProtection;
    }

    public double getMinSpeechProbability() {
        return minSpeechProbability;
    }

    public boolean isAdaptDuringSpeech() {
        return adaptDuringSpeech;
    }

    public boolean isFreezeProfile() {
        return freezeProfile;
    }

    public double getArtifactProtection() {
        return artifactProtection;
    }

    public double getAttackMs() {
        return attackMs;
    }

    public double getReleaseMs() {
        return releaseMs;
    }

    private static double clamp(double value, double min, double max) {
        if (Double.isNaN(value)) {
            return min;
        }
        return value < min ? min : (value > max ? max : value);
    }
}
