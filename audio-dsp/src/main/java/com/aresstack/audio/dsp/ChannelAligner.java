package com.aresstack.audio.dsp;

/**
 * Time-align microphone channels. Estimates the delay of each channel relative to a reference by
 * cross-correlation (with parabolic sub-sample refinement) and applies integer or fractional (linearly
 * interpolated) delays. Channel count is preserved; alignment is performed on a copy so the shift never
 * reads already-shifted samples.
 */
public final class ChannelAligner {

    private ChannelAligner() {
    }

    /**
     * @return the fractional delay (in samples) by which {@code channel} lags behind {@code reference}
     *         (positive means the channel arrives later); searched within +/- maxLag.
     */
    public static double estimateDelay(short[] samples, int count, int channels, int reference, int channel,
                                       int maxLag) {
        int ch = Math.max(1, channels);
        int frames = count / ch;
        if (frames <= 2 || reference == channel) {
            return 0.0d;
        }
        int limit = Math.max(1, Math.min(maxLag, frames - 1));
        int bestLag = 0;
        double bestValue = Double.NEGATIVE_INFINITY;
        double[] values = new double[2 * limit + 1];
        for (int lag = -limit; lag <= limit; lag++) {
            double sum = 0.0d;
            for (int t = 0; t < frames; t++) {
                int j = t + lag;
                if (j < 0 || j >= frames) {
                    continue;
                }
                sum += (double) samples[t * ch + reference] * samples[j * ch + channel];
            }
            values[lag + limit] = sum;
            if (sum > bestValue) {
                bestValue = sum;
                bestLag = lag;
            }
        }
        // Parabolic interpolation around the integer peak for sub-sample accuracy.
        int idx = bestLag + limit;
        double refined = bestLag;
        if (idx > 0 && idx < values.length - 1) {
            double ym1 = values[idx - 1];
            double y0 = values[idx];
            double yp1 = values[idx + 1];
            double denom = ym1 - 2.0d * y0 + yp1;
            if (Math.abs(denom) > 1.0e-9d) {
                refined = bestLag + 0.5d * (ym1 - yp1) / denom;
            }
        }
        return refined;
    }

    /**
     * Apply per-channel delays (positive delays a channel, negative advances it) in place, using linear
     * interpolation for the fractional part. {@code delays} is indexed by channel; missing entries are 0.
     */
    public static void applyDelays(short[] samples, int count, int channels, double[] delays) {
        int ch = Math.max(1, channels);
        int frames = count / ch;
        double[] channel = new double[frames];
        for (int c = 0; c < ch; c++) {
            double d = delays != null && c < delays.length ? delays[c] : 0.0d;
            if (d == 0.0d) {
                continue;
            }
            for (int t = 0; t < frames; t++) {
                channel[t] = samples[t * ch + c];
            }
            for (int t = 0; t < frames; t++) {
                double src = t - d; // output at t comes from input at t-d
                int i0 = (int) Math.floor(src);
                double frac = src - i0;
                double v = interp(channel, i0, frames) * (1.0d - frac) + interp(channel, i0 + 1, frames) * frac;
                samples[t * ch + c] = clamp(v);
            }
        }
    }

    private static double interp(double[] data, int index, int frames) {
        if (index < 0 || index >= frames) {
            return 0.0d;
        }
        return data[index];
    }

    private static short clamp(double value) {
        if (Double.isNaN(value)) {
            return 0;
        }
        if (value > Short.MAX_VALUE) {
            return Short.MAX_VALUE;
        }
        if (value < Short.MIN_VALUE) {
            return Short.MIN_VALUE;
        }
        return (short) Math.round(value);
    }
}
