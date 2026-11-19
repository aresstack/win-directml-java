package com.aresstack.audio.dsp;

/** Validated, defensive settings for {@link AdaptiveHumRemovalProcessor}; out-of-range input is clamped. */
public final class AdaptiveHumRemovalSettings {

    private final double baseFrequencyHz;
    private final double searchRangeHz;
    private final double adaptationSpeed;
    private final int harmonics;
    private final double maxAttenuationDb;
    private final boolean speechProtection;

    public AdaptiveHumRemovalSettings(double baseFrequencyHz, double searchRangeHz, double adaptationSpeed,
                                      int harmonics, double maxAttenuationDb, boolean speechProtection) {
        this.baseFrequencyHz = clamp(baseFrequencyHz, 20.0d, 500.0d, 50.0d);
        this.searchRangeHz = clamp(searchRangeHz, 0.0d, 20.0d, 3.0d);
        this.adaptationSpeed = clamp(adaptationSpeed, 0.0d, 1.0d, 0.1d);
        this.harmonics = (int) Math.round(clamp(harmonics, 1.0d, 12.0d, 3.0d));
        this.maxAttenuationDb = clamp(maxAttenuationDb, 0.0d, 80.0d, 24.0d);
        this.speechProtection = speechProtection;
    }

    public double getBaseFrequencyHz() {
        return baseFrequencyHz;
    }

    public double getSearchRangeHz() {
        return searchRangeHz;
    }

    public double getAdaptationSpeed() {
        return adaptationSpeed;
    }

    public int getHarmonics() {
        return harmonics;
    }

    public double getMaxAttenuationDb() {
        return maxAttenuationDb;
    }

    public boolean isSpeechProtection() {
        return speechProtection;
    }

    private static double clamp(double value, double min, double max, double fallback) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return fallback;
        }
        return value < min ? min : (value > max ? max : value);
    }
}
