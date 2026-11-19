package com.aresstack.audio.dsp;

/** Validated, defensive settings for {@link SilenceTrimmer}. Out-of-range input is clamped to safe values. */
public final class SilenceTrimmerSettings {

    private final boolean trimLeading;
    private final boolean trimTrailing;
    private final double minSpeechProbability;
    private final double preRollMs;
    private final double postRollMs;
    private final double minRetainedMs;
    private final SilenceTrimNoSpeechBehavior noSpeechBehavior;
    private final boolean zeroCrossingAlignment;
    private final double zeroCrossingSearchMs;

    public SilenceTrimmerSettings(boolean trimLeading, boolean trimTrailing, double minSpeechProbability,
                                  double preRollMs, double postRollMs, double minRetainedMs,
                                  SilenceTrimNoSpeechBehavior noSpeechBehavior, boolean zeroCrossingAlignment,
                                  double zeroCrossingSearchMs) {
        this.trimLeading = trimLeading;
        this.trimTrailing = trimTrailing;
        this.minSpeechProbability = clamp(minSpeechProbability, 0.0d, 1.0d, 0.5d);
        this.preRollMs = clamp(preRollMs, 0.0d, 5000.0d, 200.0d);
        this.postRollMs = clamp(postRollMs, 0.0d, 5000.0d, 350.0d);
        this.minRetainedMs = clamp(minRetainedMs, 0.0d, 60000.0d, 400.0d);
        this.noSpeechBehavior = noSpeechBehavior == null ? SilenceTrimNoSpeechBehavior.KEEP_ORIGINAL
                : noSpeechBehavior;
        this.zeroCrossingAlignment = zeroCrossingAlignment;
        this.zeroCrossingSearchMs = clamp(zeroCrossingSearchMs, 0.0d, 100.0d, 5.0d);
    }

    public boolean isTrimLeading() {
        return trimLeading;
    }

    public boolean isTrimTrailing() {
        return trimTrailing;
    }

    public double getMinSpeechProbability() {
        return minSpeechProbability;
    }

    public double getPreRollMs() {
        return preRollMs;
    }

    public double getPostRollMs() {
        return postRollMs;
    }

    public double getMinRetainedMs() {
        return minRetainedMs;
    }

    public SilenceTrimNoSpeechBehavior getNoSpeechBehavior() {
        return noSpeechBehavior;
    }

    public boolean isZeroCrossingAlignment() {
        return zeroCrossingAlignment;
    }

    public double getZeroCrossingSearchMs() {
        return zeroCrossingSearchMs;
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
