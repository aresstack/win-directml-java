package com.aresstack.audio.dsp;

/**
 * Smoothly track a moving speaker's direction from a stream of per-block {@link DirectionEstimate}s. Low
 * confidence does not cause abrupt jumps: the last valid direction is held for a while and, if uncertainty
 * persists, the tracker falls back to a configured direction. Accepted updates are rate-limited to a maximum
 * angular speed and then smoothed. Stateful; construct one per run.
 */
public final class DirectionTracker {

    private final double smoothing;
    private final double maxAngularSpeedDegPerBlock;
    private final double minConfidence;
    private final int holdBlocks;
    private final double fallbackAzimuthDeg;
    private final boolean updateDuringSilence;

    private double currentAzimuthDeg;
    private boolean initialized;
    private int blocksSinceValid;

    public DirectionTracker(double smoothing, double maxAngularSpeedDegPerBlock, double minConfidence,
                            int holdBlocks, double fallbackAzimuthDeg, boolean updateDuringSilence) {
        this.smoothing = clamp01(smoothing);
        this.maxAngularSpeedDegPerBlock = Math.max(0.0d, maxAngularSpeedDegPerBlock);
        this.minConfidence = clamp01(minConfidence);
        this.holdBlocks = Math.max(0, holdBlocks);
        this.fallbackAzimuthDeg = fallbackAzimuthDeg;
        this.updateDuringSilence = updateDuringSilence;
        this.currentAzimuthDeg = fallbackAzimuthDeg;
    }

    /** Feed one block's estimate and return the tracked azimuth to steer toward for that block. */
    public double update(DirectionEstimate estimate) {
        boolean valid = estimate != null && estimate.getConfidence() >= minConfidence
                && (updateDuringSilence || estimate.isSpeechActive());
        if (!initialized) {
            currentAzimuthDeg = valid ? estimate.getAzimuthDeg() : fallbackAzimuthDeg;
            initialized = true;
            blocksSinceValid = valid ? 0 : holdBlocks + 1;
            return currentAzimuthDeg;
        }
        double target;
        if (valid) {
            blocksSinceValid = 0;
            double desired = estimate.getAzimuthDeg();
            double delta = desired - currentAzimuthDeg;
            if (delta > maxAngularSpeedDegPerBlock) {
                delta = maxAngularSpeedDegPerBlock;
            } else if (delta < -maxAngularSpeedDegPerBlock) {
                delta = -maxAngularSpeedDegPerBlock;
            }
            target = currentAzimuthDeg + delta;
        } else {
            blocksSinceValid++;
            target = blocksSinceValid <= holdBlocks ? currentAzimuthDeg : fallbackAzimuthDeg;
        }
        currentAzimuthDeg += (1.0d - smoothing) * (target - currentAzimuthDeg);
        return currentAzimuthDeg;
    }

    public double getCurrentAzimuthDeg() {
        return currentAzimuthDeg;
    }

    private static double clamp01(double value) {
        if (Double.isNaN(value) || value < 0.0d) {
            return 0.0d;
        }
        return value > 1.0d ? 1.0d : value;
    }
}
