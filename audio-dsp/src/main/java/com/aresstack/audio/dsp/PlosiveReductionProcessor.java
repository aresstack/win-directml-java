package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Reduce low-frequency plosive bursts (P/B pops). A low-pass isolates the low band; a fast-versus-slow
 * envelope on that band detects a sudden low-frequency transient, and the low band is ducked by up to
 * {@code strength} during it, leaving steady low-frequency content and the rest of the spectrum intact.
 * Time-domain, built on the shared biquad; a fresh instance per run keeps state isolated.
 */
public final class PlosiveReductionProcessor {

    private static final double EPS = 1.0e-9d;
    private static final double ONSET_RATIO = 2.0d;   // fast/slow ratio where ducking begins
    private static final double RATIO_RANGE = 4.0d;   // ratio span over which ducking reaches full strength

    private final PlosiveReductionSettings settings;

    private BiquadCoefficients lowCoefficients;
    private BiquadFilterState[] lowStates;
    private double fastEnv;
    private double slowEnv;
    private double gain;
    private int configuredRate;
    private int configuredChannels;
    private boolean bypass;

    public PlosiveReductionProcessor(PlosiveReductionSettings settings) {
        if (settings == null) {
            throw new IllegalArgumentException("Plosive-reduction settings must not be null.");
        }
        this.settings = settings;
        this.gain = 1.0d;
    }

    public void process(short[] samples, int count, PcmAudioFormat format) {
        int channels = Math.max(1, format.getChannels());
        int rate = format.getSampleRateHz();
        configure(rate, channels);
        if (bypass) {
            return;
        }
        double fastCoeff = coefficient(3.0d, rate);
        double slowCoeff = coefficient(200.0d, rate);
        double attackCoeff = coefficient(settings.getAttackMs(), rate);
        double releaseCoeff = coefficient(settings.getReleaseMs(), rate);

        int frames = count / channels;
        double[] low = new double[channels];
        for (int f = 0; f < frames; f++) {
            int base = f * channels;
            double peak = 0.0d;
            for (int c = 0; c < channels; c++) {
                low[c] = lowStates[c].process(lowCoefficients, samples[base + c]);
                double abs = Math.abs(low[c]);
                if (abs > peak) {
                    peak = abs;
                }
            }
            fastEnv += fastCoeff * (peak - fastEnv);
            slowEnv += slowCoeff * (peak - slowEnv);
            double ratio = fastEnv / (slowEnv + EPS);
            double activation = clamp01((ratio - ONSET_RATIO) / RATIO_RANGE);
            double targetGain = 1.0d - settings.getStrength() * activation;
            double coeff = targetGain < gain ? attackCoeff : releaseCoeff; // more ducking = attack
            gain += coeff * (targetGain - gain);
            double lowGain = gain - 1.0d; // apply the reduction only to the low band
            for (int c = 0; c < channels; c++) {
                samples[base + c] = clamp(samples[base + c] + low[c] * lowGain);
            }
        }
    }

    private void configure(int rate, int channels) {
        if (lowStates != null && configuredRate == rate && configuredChannels == channels) {
            return;
        }
        configuredRate = rate;
        configuredChannels = channels;
        try {
            lowCoefficients = BiquadCoefficients.lowPass(rate, settings.getTargetFrequencyHz(), 0.707d);
            bypass = false;
        } catch (RuntimeException invalid) {
            bypass = true;
            return;
        }
        lowStates = new BiquadFilterState[channels];
        for (int c = 0; c < channels; c++) {
            lowStates[c] = new BiquadFilterState();
        }
        fastEnv = 0.0d;
        slowEnv = 0.0d;
        gain = 1.0d;
    }

    private static double coefficient(double milliseconds, int rate) {
        double samples = Math.max(1.0d, milliseconds * rate / 1000.0d);
        return 1.0d - Math.exp(-1.0d / samples);
    }

    private static double clamp01(double value) {
        if (Double.isNaN(value) || value < 0.0d) {
            return 0.0d;
        }
        return value > 1.0d ? 1.0d : value;
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
