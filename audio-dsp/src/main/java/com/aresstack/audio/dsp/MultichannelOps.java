package com.aresstack.audio.dsp;

/**
 * Small, pure channel-domain operations shared by the multichannel blocks: selecting a channel, weighted
 * downmix to mono, per-channel gain/polarity, and inter-channel correlation. All operate on interleaved
 * PCM16 and never touch Swing/AWT.
 */
public final class MultichannelOps {

    private MultichannelOps() {
    }

    /** @return a mono copy consisting of channel {@code index} (clamped into range). */
    public static short[] selectChannel(short[] samples, int count, int channels, int index) {
        int ch = Math.max(1, channels);
        int pick = index < 0 ? 0 : (index >= ch ? ch - 1 : index);
        int frames = count / ch;
        short[] out = new short[frames];
        for (int f = 0; f < frames; f++) {
            out[f] = samples[f * ch + pick];
        }
        return out;
    }

    /**
     * Weighted downmix to mono: {@code out[f] = sum_i weight[i] * x[f, i]}. A negative weight inverts that
     * input's polarity. When {@code normalize} is set the sum is divided by the total absolute weight.
     */
    public static short[] downmixToMono(short[] samples, int count, int channels, double[] weights,
                                        boolean normalize) {
        int ch = Math.max(1, channels);
        int frames = count / ch;
        double norm = 1.0d;
        if (normalize) {
            double sum = 0.0d;
            for (int i = 0; i < ch; i++) {
                sum += Math.abs(weightOf(weights, i, ch));
            }
            norm = sum > 1.0e-9d ? sum : 1.0d;
        }
        short[] out = new short[frames];
        for (int f = 0; f < frames; f++) {
            double acc = 0.0d;
            int base = f * ch;
            for (int i = 0; i < ch; i++) {
                acc += weightOf(weights, i, ch) * samples[base + i];
            }
            out[f] = clamp(acc / norm);
        }
        return out;
    }

    /** Apply per-channel linear gain and optional polarity inversion in place (channel count preserved). */
    public static void applyGainPolarity(short[] samples, int count, int channels, double[] gains,
                                         boolean[] invert) {
        int ch = Math.max(1, channels);
        int frames = count / ch;
        for (int f = 0; f < frames; f++) {
            int base = f * ch;
            for (int i = 0; i < ch; i++) {
                double g = gains != null && i < gains.length ? gains[i] : 1.0d;
                if (invert != null && i < invert.length && invert[i]) {
                    g = -g;
                }
                samples[base + i] = clamp(samples[base + i] * g);
            }
        }
    }

    /** @return the normalized cross-correlation (Pearson) between two channels, in [-1, 1]. */
    public static double correlation(short[] samples, int count, int channels, int a, int b) {
        int ch = Math.max(1, channels);
        if (a < 0 || b < 0 || a >= ch || b >= ch) {
            return 0.0d;
        }
        int frames = count / ch;
        double sumAB = 0.0d;
        double sumAA = 0.0d;
        double sumBB = 0.0d;
        for (int f = 0; f < frames; f++) {
            double va = samples[f * ch + a];
            double vb = samples[f * ch + b];
            sumAB += va * vb;
            sumAA += va * va;
            sumBB += vb * vb;
        }
        double denom = Math.sqrt(sumAA * sumBB);
        return denom > 1.0e-9d ? sumAB / denom : 0.0d;
    }

    private static double weightOf(double[] weights, int i, int channels) {
        if (weights == null || weights.length == 0) {
            return 1.0d / channels; // default: equal average
        }
        return i < weights.length ? weights[i] : 0.0d;
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
