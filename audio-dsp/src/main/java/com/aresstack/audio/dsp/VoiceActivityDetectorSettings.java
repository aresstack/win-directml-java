package com.aresstack.audio.dsp;

/**
 * Validated, defensive settings for {@link VoiceActivityDetector}. Every value is clamped to a sane range
 * and non-finite input is replaced with a default, so the detector can never be driven into NaN/Infinity or
 * instability even if a profile carries out-of-range parameters. (The profile validator reports such values
 * to the user separately; this is the last-line runtime guard.)
 */
public final class VoiceActivityDetectorSettings {

    private final double sensitivity;
    private final double minSpeechProbability;
    private final int frameDurationMs;
    private final double attackMs;
    private final double releaseMs;
    private final double hangoverMs;
    private final double minSpeechMs;
    private final double minSilenceMs;
    private final double noiseAdaptationSpeed;
    private final boolean adaptNoiseDuringSpeech;

    public VoiceActivityDetectorSettings(double sensitivity, double minSpeechProbability, int frameDurationMs,
                                         double attackMs, double releaseMs, double hangoverMs,
                                         double minSpeechMs, double minSilenceMs, double noiseAdaptationSpeed,
                                         boolean adaptNoiseDuringSpeech) {
        this.sensitivity = clamp(sensitivity, 0.0d, 1.0d, 0.5d);
        this.minSpeechProbability = clamp(minSpeechProbability, 0.0d, 1.0d, 0.5d);
        this.frameDurationMs = (int) Math.round(clamp(frameDurationMs, 5.0d, 60.0d, 20.0d));
        this.attackMs = clamp(attackMs, 0.0d, 5000.0d, 50.0d);
        this.releaseMs = clamp(releaseMs, 0.0d, 10000.0d, 300.0d);
        this.hangoverMs = clamp(hangoverMs, 0.0d, 10000.0d, 200.0d);
        this.minSpeechMs = clamp(minSpeechMs, 0.0d, 5000.0d, 80.0d);
        this.minSilenceMs = clamp(minSilenceMs, 0.0d, 10000.0d, 150.0d);
        this.noiseAdaptationSpeed = clamp(noiseAdaptationSpeed, 0.0001d, 0.5d, 0.05d);
        this.adaptNoiseDuringSpeech = adaptNoiseDuringSpeech;
    }

    public double getSensitivity() {
        return sensitivity;
    }

    public double getMinSpeechProbability() {
        return minSpeechProbability;
    }

    public int getFrameDurationMs() {
        return frameDurationMs;
    }

    public double getAttackMs() {
        return attackMs;
    }

    public double getReleaseMs() {
        return releaseMs;
    }

    public double getHangoverMs() {
        return hangoverMs;
    }

    public double getMinSpeechMs() {
        return minSpeechMs;
    }

    public double getMinSilenceMs() {
        return minSilenceMs;
    }

    public double getNoiseAdaptationSpeed() {
        return noiseAdaptationSpeed;
    }

    public boolean isAdaptNoiseDuringSpeech() {
        return adaptNoiseDuringSpeech;
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
