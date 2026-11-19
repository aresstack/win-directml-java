package com.aresstack.audio.dsp;

/**
 * STFT de-esser: measures the energy in the sibilance band and, when it exceeds the threshold, attenuates
 * only that band's bins (phase preserved), smoothed with attack/release across frames.
 */
public final class SpectralDeEsser implements SpectralModifier {

    private final DeEsserSettings settings;
    private final double hopSamples;
    private double gainDb;

    public SpectralDeEsser(DeEsserSettings settings, double hopSamples) {
        this.settings = settings;
        this.hopSamples = hopSamples;
    }

    public void modify(double[] real, double[] imag, int sampleRateHz, int frameStartSample) {
        int n = real.length;
        int half = n / 2;
        double binHz = (double) sampleRateHz / n;
        int loBin = Spectra.binOf(settings.getTargetFrequencyHz() - settings.getBandwidthHz() / 2.0d, binHz, half);
        int hiBin = Spectra.binOf(settings.getTargetFrequencyHz() + settings.getBandwidthHz() / 2.0d, binHz, half);
        if (hiBin < loBin) {
            return;
        }
        double energy = 0.0d;
        for (int k = loBin; k <= hiBin; k++) {
            double m = Spectra.magnitude(real, imag, k);
            energy += m * m;
        }
        double bandRms = Math.sqrt(2.0d * energy) / n;
        double levelDb = 20.0d * Math.log10(Math.max(bandRms, Spectra.EPS) / Spectra.FULL_SCALE);
        double reduction = Math.max(0.0d, Math.min(settings.getReductionDb(), levelDb - settings.getThresholdDb()));
        double targetGainDb = -reduction;
        double hopMs = hopSamples * 1000.0d / sampleRateHz;
        double coeff = targetGainDb < gainDb
                ? Spectra.coefficient(settings.getAttackMs(), hopMs)
                : Spectra.coefficient(settings.getReleaseMs(), hopMs);
        gainDb += coeff * (targetGainDb - gainDb);
        double g = Math.pow(10.0d, gainDb / 20.0d);
        for (int k = loBin; k <= hiBin; k++) {
            Spectra.applyGain(real, imag, k, g);
        }
    }
}
