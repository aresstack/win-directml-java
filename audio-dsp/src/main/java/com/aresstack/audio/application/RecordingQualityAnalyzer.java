package com.aresstack.audio.application;

/**
 * Turn the raw measurements of a finished recording (duration, level statistics, dropped frame count)
 * into a single {@link RecordingQuality} verdict, using {@link RecordingQualityThresholds}. Pure and
 * stateless; the numbers come from the level meter over the actual captured signal.
 *
 * <p>Order of judgement: the blocking problems first (too short, then no signal), then clipping
 * (a warning), then dropped frames, otherwise valid.</p>
 */
public final class RecordingQualityAnalyzer {

    private final RecordingQualityThresholds thresholds;

    public RecordingQualityAnalyzer(RecordingQualityThresholds thresholds) {
        this.thresholds = thresholds == null ? RecordingQualityThresholds.defaults() : thresholds;
    }

    public static RecordingQualityAnalyzer withDefaults() {
        return new RecordingQualityAnalyzer(RecordingQualityThresholds.defaults());
    }

    /**
     * @param durationMillis recording length
     * @param overallRms     RMS across the whole recording (from the level meter)
     * @param peak           peak sample magnitude (0..32767)
     * @param clippedSamples number of samples at the 16-bit clipping limit
     * @param totalSamples   total samples measured
     * @param droppedFrames  frames the capture-to-disk stage could not keep up with
     */
    public RecordingQuality analyze(long durationMillis, double overallRms, int peak,
                                    long clippedSamples, long totalSamples, long droppedFrames) {
        if (durationMillis < thresholds.getMinDurationMillis()) {
            return RecordingQuality.TOO_SHORT;
        }
        if (overallRms < thresholds.getMinRms() && peak < thresholds.getMinPeak()) {
            return RecordingQuality.NO_SIGNAL;
        }
        double clippedFraction = totalSamples <= 0 ? 0.0d : (double) clippedSamples / (double) totalSamples;
        if (clippedFraction > thresholds.getMaxClippedFraction()) {
            return RecordingQuality.CLIPPED;
        }
        if (droppedFrames > thresholds.getMaxDroppedFrames()) {
            return RecordingQuality.DROPPED_FRAMES;
        }
        return RecordingQuality.VALID;
    }
}
