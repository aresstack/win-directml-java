package com.aresstack.audio.dsp;

/**
 * STFT plosive reduction: tracks the low-band energy and, on a sudden low-frequency transient (fast rise
 * over a slow average), ducks the low-band bins by up to {@code strength}, phase preserved.
 */
public final class SpectralPlosiveReduction implements SpectralModifier {

    private static final double ONSET_RATIO = 2.0d;
    private static final double RATIO_RANGE = 4.0d;

    private final PlosiveReductionSettings settings;
    private final double hopSamples;
    private double slowEnergy;
    private double gain = 1.0d;
    private boolean initialized;

    public SpectralPlosiveReduction(PlosiveReductionSettings settings, double hopSamples) {
        this.settings = settings;
        this.hopSamples = hopSamples;
    }

    public void modify(double[] real, double[] imag, int sampleRateHz, int frameStartSample) {
        int n = real.length;
        int half = n / 2;
        double binHz = (double) sampleRateHz / n;
        int topBin = Spectra.binOf(settings.getTargetFrequencyHz(), binHz, half);
        double energy = 0.0d;
        for (int k = 1; k <= topBin; k++) {
            double m = Spectra.magnitude(real, imag, k);
            energy += m * m;
        }
        if (!initialized) {
            slowEnergy = energy;
            initialized = true;
        }
        double hopMs = hopSamples * 1000.0d / sampleRateHz;
        double slowCoeff = Spectra.coefficient(200.0d, hopMs);
        double ratio = energy / (slowEnergy + Spectra.EPS);
        slowEnergy += slowCoeff * (energy - slowEnergy);
        double activation = Spectra.clamp01((ratio - ONSET_RATIO) / RATIO_RANGE);
        double targetGain = 1.0d - settings.getStrength() * activation;
        double coeff = targetGain < gain
                ? Spectra.coefficient(settings.getAttackMs(), hopMs)
                : Spectra.coefficient(settings.getReleaseMs(), hopMs);
        gain += coeff * (targetGain - gain);
        for (int k = 1; k <= topBin; k++) {
            Spectra.applyGain(real, imag, k, gain);
        }
    }
}
