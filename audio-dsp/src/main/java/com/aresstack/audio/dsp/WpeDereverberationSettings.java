package com.aresstack.audio.dsp;

/** Immutable configuration for {@link WpeDereverberation}. Values are clamped to sane ranges. */
public final class WpeDereverberationSettings {

    /** Processing mode. Offline processes the whole signal at once; block-adaptive/streaming re-estimate. */
    public enum Mode {
        OFFLINE,
        BLOCK_ADAPTIVE,
        STREAMING
    }

    private final Mode mode;
    private final double strength;
    private final int predictionDelay;
    private final int filterLength;
    private final int iterations;
    private final double earlyReflectionPreservation;
    private final boolean speechProtection;
    private final double artifactProtection;
    private final double adaptationSpeed;
    private final int blockSizeFrames;

    public WpeDereverberationSettings(Mode mode, double strength, int predictionDelay, int filterLength,
                                      int iterations, double earlyReflectionPreservation,
                                      boolean speechProtection, double artifactProtection,
                                      double adaptationSpeed, int blockSizeFrames) {
        this.mode = mode == null ? Mode.OFFLINE : mode;
        this.strength = clamp(strength, 0.0d, 1.0d);
        this.predictionDelay = clampInt(predictionDelay, 1, 32);
        this.filterLength = clampInt(filterLength, 1, 32);
        this.iterations = clampInt(iterations, 1, 10);
        this.earlyReflectionPreservation = clamp(earlyReflectionPreservation, 0.0d, 1.0d);
        this.speechProtection = speechProtection;
        this.artifactProtection = clamp(artifactProtection, 0.0d, 1.0d);
        this.adaptationSpeed = clamp(adaptationSpeed, 0.0d, 1.0d);
        this.blockSizeFrames = clampInt(blockSizeFrames, 8, 4096);
    }

    public Mode getMode() {
        return mode;
    }

    public double getStrength() {
        return strength;
    }

    /** @return the base prediction delay in frames, plus the early-reflection preservation offset. */
    public int getEffectivePredictionDelay() {
        return predictionDelay + (int) Math.round(earlyReflectionPreservation * 3.0d);
    }

    public int getFilterLength() {
        return filterLength;
    }

    public int getIterations() {
        return iterations;
    }

    public double getEarlyReflectionPreservation() {
        return earlyReflectionPreservation;
    }

    public boolean isSpeechProtection() {
        return speechProtection;
    }

    public double getArtifactProtection() {
        return artifactProtection;
    }

    public double getAdaptationSpeed() {
        return adaptationSpeed;
    }

    public int getBlockSizeFrames() {
        return blockSizeFrames;
    }

    private static double clamp(double value, double min, double max) {
        if (Double.isNaN(value)) {
            return min;
        }
        return value < min ? min : (value > max ? max : value);
    }

    private static int clampInt(int value, int min, int max) {
        return value < min ? min : (value > max ? max : value);
    }
}
