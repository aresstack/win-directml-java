package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * A split-band de-esser: a band-pass isolates the sibilance band, its (channel-linked) level drives a
 * dynamic attenuation, and only that band is reduced — the rest of the signal passes through. This keeps
 * the voice intact while taming over-emphasized S/sh sounds. Time-domain, built on the shared biquad.
 *
 * <p>A fresh instance is created per pipeline run (see the registry factory), so the internal filter and
 * envelope state never leaks between recordings.</p>
 */
public final class DeEsserProcessor {

    private static final double FULL_SCALE = 32768.0d;
    private static final double EPS = 1.0e-9d;

    private final DeEsserSettings settings;

    private BiquadCoefficients bandCoefficients;
    private BiquadFilterState[] bandStates;
    private double smoothedBand;
    private double gainDb;
    private int configuredRate;
    private int configuredChannels;
    private boolean bypass;

    public DeEsserProcessor(DeEsserSettings settings) {
        if (settings == null) {
            throw new IllegalArgumentException("De-esser settings must not be null.");
        }
        this.settings = settings;
    }

    public void process(short[] samples, int count, PcmAudioFormat format) {
        int channels = Math.max(1, format.getChannels());
        int rate = format.getSampleRateHz();
        configure(rate, channels);
        if (bypass) {
            return;
        }
        double envCoeff = coefficient(5.0d, rate); // ~5 ms detector smoothing
        double attackCoeff = coefficient(settings.getAttackMs(), rate);
        double releaseCoeff = coefficient(settings.getReleaseMs(), rate);

        int frames = count / channels;
        double[] band = new double[channels];
        for (int f = 0; f < frames; f++) {
            int base = f * channels;
            double peak = 0.0d;
            for (int c = 0; c < channels; c++) {
                band[c] = bandStates[c].process(bandCoefficients, samples[base + c]);
                double abs = Math.abs(band[c]);
                if (abs > peak) {
                    peak = abs;
                }
            }
            smoothedBand += envCoeff * (peak - smoothedBand);
            double levelDb = 20.0d * Math.log10(Math.max(smoothedBand, EPS) / FULL_SCALE);
            double excess = levelDb - settings.getThresholdDb();
            double targetGainDb = -Math.max(0.0d, Math.min(settings.getReductionDb(), excess));
            double coeff = targetGainDb < gainDb ? attackCoeff : releaseCoeff; // more reduction = attack
            gainDb += coeff * (targetGainDb - gainDb);
            double bandGain = Math.pow(10.0d, gainDb / 20.0d) - 1.0d; // apply only to the band
            for (int c = 0; c < channels; c++) {
                samples[base + c] = clamp(samples[base + c] + band[c] * bandGain);
            }
        }
    }

    private void configure(int rate, int channels) {
        if (bandStates != null && configuredRate == rate && configuredChannels == channels) {
            return;
        }
        configuredRate = rate;
        configuredChannels = channels;
        double q = settings.getTargetFrequencyHz() / Math.max(1.0d, settings.getBandwidthHz());
        try {
            bandCoefficients = BiquadCoefficients.bandPass(rate, settings.getTargetFrequencyHz(),
                    Math.max(0.1d, q));
            bypass = false;
        } catch (RuntimeException invalid) {
            bypass = true; // target band not valid for this rate: pass audio through unchanged
            return;
        }
        bandStates = new BiquadFilterState[channels];
        for (int c = 0; c < channels; c++) {
            bandStates[c] = new BiquadFilterState();
        }
        smoothedBand = 0.0d;
        gainDb = 0.0d;
    }

    private static double coefficient(double milliseconds, int rate) {
        double samples = Math.max(1.0d, milliseconds * rate / 1000.0d);
        return 1.0d - Math.exp(-1.0d / samples);
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
