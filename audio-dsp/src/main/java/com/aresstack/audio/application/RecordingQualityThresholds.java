package com.aresstack.audio.application;

/**
 * Central, immutable thresholds for {@link RecordingQualityAnalyzer}. Kept in one place so the limits
 * are not scattered across UI code and can be tuned or overridden in one spot.
 */
public final class RecordingQualityThresholds {

    private final long minDurationMillis;
    private final double minRms;
    private final int minPeak;
    private final double maxClippedFraction;
    private final long maxDroppedFrames;

    public RecordingQualityThresholds(long minDurationMillis, double minRms, int minPeak,
                                      double maxClippedFraction, long maxDroppedFrames) {
        this.minDurationMillis = minDurationMillis;
        this.minRms = minRms;
        this.minPeak = minPeak;
        this.maxClippedFraction = maxClippedFraction;
        this.maxDroppedFrames = maxDroppedFrames;
    }

    /**
     * Defaults tuned for 16-bit speech: at least 300 ms, an overall RMS above ~30 and a peak above 500
     * to count as signal, more than 0.5% clipped samples counts as clipping, and any dropped frame is
     * reported.
     */
    public static RecordingQualityThresholds defaults() {
        return new RecordingQualityThresholds(300L, 30.0d, 500, 0.005d, 0L);
    }

    public long getMinDurationMillis() {
        return minDurationMillis;
    }

    public double getMinRms() {
        return minRms;
    }

    public int getMinPeak() {
        return minPeak;
    }

    public double getMaxClippedFraction() {
        return maxClippedFraction;
    }

    public long getMaxDroppedFrames() {
        return maxDroppedFrames;
    }
}
