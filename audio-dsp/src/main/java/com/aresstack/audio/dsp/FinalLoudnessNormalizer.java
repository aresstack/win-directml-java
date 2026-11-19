package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Offline final loudness normalization: measure the whole processed result once (target RMS or peak), then
 * apply one constant gain so the file reaches a defined end level. Because the gain is constant across the
 * whole signal, no dynamic pumping is introduced within the file. The applied gain is bounded by the maximum
 * total boost/cut, respects the peak ceiling (so it never clips), and can be restricted to attenuation-only
 * or amplification-only. Integrated loudness (LUFS) is deliberately not implemented here.
 */
public final class FinalLoudnessNormalizer {

    private static final double FULL_SCALE = 32768.0d;
    private static final double EPS = 1.0e-9d;

    private final FinalLoudnessNormalizerSettings settings;

    public FinalLoudnessNormalizer(FinalLoudnessNormalizerSettings settings) {
        if (settings == null) {
            throw new IllegalArgumentException("Loudness normalizer settings must not be null.");
        }
        this.settings = settings;
    }

    public void process(short[] samples, int count) {
        if (count <= 0) {
            return;
        }
        double peak = 0.0d;
        double sumSquares = 0.0d;
        for (int i = 0; i < count; i++) {
            double v = samples[i];
            sumSquares += v * v;
            double abs = Math.abs(v);
            if (abs > peak) {
                peak = abs;
            }
        }
        double rms = Math.sqrt(sumSquares / count);
        double measured = settings.getMode() == FinalLoudnessNormalizerSettings.Mode.PEAK ? peak : rms;
        double measuredDb = 20.0d * Math.log10(Math.max(measured, EPS) / FULL_SCALE);

        double gainDb = settings.getTargetLevelDb() - measuredDb;
        if (gainDb > 0.0d && !settings.isAllowAmplification()) {
            gainDb = 0.0d;
        }
        if (gainDb < 0.0d && !settings.isAllowAttenuation()) {
            gainDb = 0.0d;
        }
        gainDb = clamp(gainDb, -settings.getMaxTotalAttenuationDb(), settings.getMaxTotalGainDb());

        if (settings.isClippingProtection() && peak > 0.0d) {
            double ceilingAmp = Math.pow(10.0d, settings.getPeakCeilingDb() / 20.0d) * FULL_SCALE;
            double maxGainNoClipDb = 20.0d * Math.log10(ceilingAmp / peak);
            if (gainDb > maxGainNoClipDb) {
                gainDb = maxGainNoClipDb;
            }
        }
        if (Math.abs(gainDb) < 1.0e-4d) {
            return;
        }
        double linear = Math.pow(10.0d, gainDb / 20.0d);
        for (int i = 0; i < count; i++) {
            samples[i] = clamp(samples[i] * linear);
        }
    }

    private static double clamp(double value, double min, double max) {
        if (Double.isNaN(value)) {
            return 0.0d;
        }
        return value < min ? min : (value > max ? max : value);
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
