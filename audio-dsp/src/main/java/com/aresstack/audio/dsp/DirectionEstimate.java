package com.aresstack.audio.dsp;

/**
 * A direction-of-arrival estimate: azimuth and (optional) elevation in degrees, a confidence in [0, 1] and
 * whether speech was active in the analyzed window. Produced by the direction-of-arrival estimator and used
 * to steer the beamformer toward a moving speaker.
 */
public final class DirectionEstimate {

    private final double azimuthDeg;
    private final double elevationDeg;
    private final double confidence;
    private final boolean speechActive;

    public DirectionEstimate(double azimuthDeg, double elevationDeg, double confidence, boolean speechActive) {
        this.azimuthDeg = azimuthDeg;
        this.elevationDeg = elevationDeg;
        this.confidence = clamp01(confidence);
        this.speechActive = speechActive;
    }

    public double getAzimuthDeg() {
        return azimuthDeg;
    }

    public double getElevationDeg() {
        return elevationDeg;
    }

    public double getConfidence() {
        return confidence;
    }

    public boolean isSpeechActive() {
        return speechActive;
    }

    private static double clamp01(double value) {
        if (Double.isNaN(value) || value < 0.0d) {
            return 0.0d;
        }
        return value > 1.0d ? 1.0d : value;
    }
}
