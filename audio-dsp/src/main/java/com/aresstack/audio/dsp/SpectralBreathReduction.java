package com.aresstack.audio.dsp;

/**
 * STFT breath reduction: uses spectral flatness (a noise-like, low-tonal spectrum indicates breath) and,
 * with speech protection, the upstream speech gate, to attenuate suspected breath frames broadband by up to
 * the maximum attenuation, scaled by sensitivity and smoothed across frames.
 */
public final class SpectralBreathReduction implements SpectralModifier {

    private static final double FLATNESS_RANGE = 0.4d;

    private final BreathReductionSettings settings;
    private final double hopSamples;
    private final SpeechGate speechGate;
    private double gainDb;

    public SpectralBreathReduction(BreathReductionSettings settings, double hopSamples, SpeechGate speechGate) {
        this.settings = settings;
        this.hopSamples = hopSamples;
        this.speechGate = speechGate == null ? SpeechGate.NEVER : speechGate;
    }

    public void modify(double[] real, double[] imag, int sampleRateHz, int frameStartSample) {
        int half = real.length / 2;
        double logSum = 0.0d;
        double linearSum = 0.0d;
        int bins = 0;
        for (int k = 1; k < half; k++) {
            double m = Spectra.magnitude(real, imag, k) + Spectra.EPS;
            logSum += Math.log(m);
            linearSum += m;
            bins++;
        }
        double targetGainDb = 0.0d;
        boolean speech = settings.isSpeechProtection()
                && speechGate.isSpeech(frameStartSample + real.length / 2);
        if (bins > 0 && !speech) {
            double geoMean = Math.exp(logSum / bins);
            double ariMean = linearSum / bins;
            double flatness = geoMean / (ariMean + Spectra.EPS);
            double threshold = 0.5d - settings.getSensitivity() * 0.3d; // higher sensitivity → lower bar
            double activation = Spectra.clamp01((flatness - threshold) / FLATNESS_RANGE);
            targetGainDb = -settings.getMaxAttenuationDb() * activation;
        }
        double hopMs = hopSamples * 1000.0d / sampleRateHz;
        double coeff = targetGainDb < gainDb
                ? Spectra.coefficient(settings.getAttackMs(), hopMs)
                : Spectra.coefficient(settings.getReleaseMs(), hopMs);
        gainDb += coeff * (targetGainDb - gainDb);
        double g = Math.pow(10.0d, gainDb / 20.0d);
        for (int k = 1; k <= half; k++) {
            Spectra.applyGain(real, imag, k, g);
        }
    }
}
