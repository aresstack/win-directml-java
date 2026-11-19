package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Attenuate (never mute) frames whose RMS falls below a threshold, so background hiss between
 * words gets quieter without the hard on/off pumping of a classic gate. Smooth the gain with
 * separate attack and release times to avoid clicks; keep the gain across frames.
 */
public final class SoftNoiseGateProcessor implements Pcm16Processor {

    private final double thresholdRms;
    private final double closedGain;
    private final double attackMillis;
    private final double releaseMillis;

    private double currentGain = 1.0d;

    /**
     * @param thresholdRms  RMS level (in sample units, 0..32767) below which the gate closes
     * @param closedGain    gain applied when closed, e.g. 0.25 — reduce, do not mute
     * @param attackMillis  time to reopen when speech starts (keep short, e.g. 5 ms)
     * @param releaseMillis time to close after speech ends (keep long, e.g. 150 ms)
     */
    public SoftNoiseGateProcessor(double thresholdRms, double closedGain,
                                  double attackMillis, double releaseMillis) {
        if (closedGain < 0 || closedGain > 1) {
            throw new IllegalArgumentException("Closed gain must be within [0, 1]: " + closedGain);
        }
        this.thresholdRms = thresholdRms;
        this.closedGain = closedGain;
        this.attackMillis = attackMillis;
        this.releaseMillis = releaseMillis;
    }

    @Override
    public void process(short[] samples, int sampleCount, PcmAudioFormat format) {
        if (sampleCount == 0) {
            return;
        }
        double targetGain = frameRms(samples, sampleCount) < thresholdRms ? closedGain : 1.0d;
        double attackCoefficient = smoothingCoefficient(attackMillis, format.getSampleRateHz());
        double releaseCoefficient = smoothingCoefficient(releaseMillis, format.getSampleRateHz());
        for (int i = 0; i < sampleCount; i++) {
            // Opening (gain rising) uses the fast attack; closing uses the slow release.
            double coefficient = targetGain > currentGain ? attackCoefficient : releaseCoefficient;
            currentGain += coefficient * (targetGain - currentGain);
            samples[i] = (short) Math.round(samples[i] * currentGain);
        }
    }

    private static double frameRms(short[] samples, int sampleCount) {
        double sumOfSquares = 0;
        for (int i = 0; i < sampleCount; i++) {
            sumOfSquares += (double) samples[i] * samples[i];
        }
        return Math.sqrt(sumOfSquares / sampleCount);
    }

    /** Convert a time constant in milliseconds into a per-sample one-pole smoothing coefficient. */
    private static double smoothingCoefficient(double millis, int sampleRateHz) {
        if (millis <= 0) {
            return 1.0d;
        }
        return 1.0d - Math.exp(-1000.0d / (millis * sampleRateHz));
    }
}
