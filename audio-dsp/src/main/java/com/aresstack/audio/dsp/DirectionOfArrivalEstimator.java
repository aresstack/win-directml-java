package com.aresstack.audio.dsp;

/**
 * Estimate the direction of arrival of a source from a two-(or more-)microphone array. It measures the
 * time difference of arrival between the first two microphones by windowed cross-correlation (with parabolic
 * sub-sample refinement) and maps it to an azimuth relative to the array axis using the known spacing. The
 * normalized correlation peak is reported as confidence. No audio is modified.
 */
public final class DirectionOfArrivalEstimator {

    private DirectionOfArrivalEstimator() {
    }

    /** Estimate the azimuth over the frame range {@code [fromFrame, toFrame)} of an interleaved buffer. */
    public static DirectionEstimate estimate(short[] samples, int fromFrame, int toFrame, int channels,
                                             MicrophoneArrayProfile array, int rate, double speedOfSoundMmPerS,
                                             int maxLag) {
        int ch = Math.max(1, channels);
        if (ch < 2 || array == null || array.getMicrophoneCount() < 2) {
            return new DirectionEstimate(0.0d, 0.0d, 0.0d, false);
        }
        double[] p0 = array.position(0);
        double[] p1 = array.position(1);
        double spacing = Math.sqrt(sq(p1[0] - p0[0]) + sq(p1[1] - p0[1]) + sq(p1[2] - p0[2]));
        if (spacing < 1.0e-6d) {
            return new DirectionEstimate(0.0d, 0.0d, 0.0d, false);
        }
        int n = toFrame - fromFrame;
        int limit = Math.max(1, Math.min(maxLag, n - 1));
        double auto0 = 0.0d;
        double auto1 = 0.0d;
        for (int t = fromFrame; t < toFrame; t++) {
            double a = samples[t * ch];
            double b = samples[t * ch + 1];
            auto0 += a * a;
            auto1 += b * b;
        }
        double denom = Math.sqrt(auto0 * auto1);
        if (denom < 1.0e-6d) {
            return new DirectionEstimate(90.0d, 0.0d, 0.0d, false); // silent window
        }
        int bestLag = 0;
        double bestValue = Double.NEGATIVE_INFINITY;
        double[] values = new double[2 * limit + 1];
        for (int lag = -limit; lag <= limit; lag++) {
            double sum = 0.0d;
            for (int t = fromFrame; t < toFrame; t++) {
                int j = t + lag;
                if (j < fromFrame || j >= toFrame) {
                    continue;
                }
                sum += (double) samples[t * ch] * samples[j * ch + 1];
            }
            values[lag + limit] = sum;
            if (sum > bestValue) {
                bestValue = sum;
                bestLag = lag;
            }
        }
        double refined = bestLag;
        int idx = bestLag + limit;
        if (idx > 0 && idx < values.length - 1) {
            double ym1 = values[idx - 1];
            double y0 = values[idx];
            double yp1 = values[idx + 1];
            double d = ym1 - 2.0d * y0 + yp1;
            if (Math.abs(d) > 1.0e-9d) {
                refined = bestLag + 0.5d * (ym1 - yp1) / d;
            }
        }
        // lag is how much channel 1 lags channel 0; cos(azimuth from axis) = -lag*c/(spacing*rate).
        double cos = -refined * speedOfSoundMmPerS / (spacing * rate);
        cos = cos < -1.0d ? -1.0d : (cos > 1.0d ? 1.0d : cos);
        double azimuth = Math.toDegrees(Math.acos(cos));
        double confidence = clamp01(bestValue / denom);
        return new DirectionEstimate(azimuth, 0.0d, confidence, true);
    }

    private static double sq(double v) {
        return v * v;
    }

    private static double clamp01(double value) {
        if (Double.isNaN(value) || value < 0.0d) {
            return 0.0d;
        }
        return value > 1.0d ? 1.0d : value;
    }
}
