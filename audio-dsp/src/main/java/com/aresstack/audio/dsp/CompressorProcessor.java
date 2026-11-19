package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Even out speech loudness with a gentle feed-forward compressor: track the signal envelope,
 * and above the threshold reduce the excess by the given ratio. Keep the envelope and gain
 * across frames. Use moderate settings — over-compression makes speech harder for STT, not
 * easier.
 */
public final class CompressorProcessor implements Pcm16Processor {

    private final double threshold;
    private final double ratio;
    private final double attackMillis;
    private final double releaseMillis;

    private double envelope;

    /**
     * @param threshold     absolute sample level (0..32767) above which compression starts
     * @param ratio         compression ratio, e.g. 3 means 3:1 above the threshold
     * @param attackMillis  how fast the compressor reacts to loud passages (e.g. 5 ms)
     * @param releaseMillis how fast it recovers after loud passages (e.g. 100 ms)
     */
    public CompressorProcessor(double threshold, double ratio, double attackMillis, double releaseMillis) {
        if (ratio < 1) {
            throw new IllegalArgumentException("Compression ratio must be >= 1: " + ratio);
        }
        this.threshold = threshold;
        this.ratio = ratio;
        this.attackMillis = attackMillis;
        this.releaseMillis = releaseMillis;
    }

    @Override
    public void process(short[] samples, int sampleCount, PcmAudioFormat format) {
        double attackCoefficient = smoothingCoefficient(attackMillis, format.getSampleRateHz());
        double releaseCoefficient = smoothingCoefficient(releaseMillis, format.getSampleRateHz());
        for (int i = 0; i < sampleCount; i++) {
            double magnitude = Math.abs((double) samples[i]);
            // Track the envelope: fast rise (attack), slow fall (release).
            double coefficient = magnitude > envelope ? attackCoefficient : releaseCoefficient;
            envelope += coefficient * (magnitude - envelope);

            double gain = 1.0d;
            if (envelope > threshold) {
                double compressedLevel = threshold + (envelope - threshold) / ratio;
                gain = compressedLevel / envelope;
            }
            samples[i] = clamp(samples[i] * gain);
        }
    }

    /** Convert a time constant in milliseconds into a per-sample one-pole smoothing coefficient. */
    private static double smoothingCoefficient(double millis, int sampleRateHz) {
        if (millis <= 0) {
            return 1.0d;
        }
        return 1.0d - Math.exp(-1000.0d / (millis * sampleRateHz));
    }

    private static short clamp(double value) {
        if (value > Short.MAX_VALUE) {
            return Short.MAX_VALUE;
        }
        if (value < Short.MIN_VALUE) {
            return Short.MIN_VALUE;
        }
        return (short) Math.round(value);
    }
}
