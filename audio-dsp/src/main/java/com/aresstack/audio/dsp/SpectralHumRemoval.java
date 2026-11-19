package com.aresstack.audio.dsp;

/**
 * STFT adaptive hum removal: locates the mains fundamental as the strongest bin within the search range,
 * tracks its drift with the adaptation speed, and attenuates the bins at the fundamental and its harmonics
 * (and their immediate neighbours) by the maximum attenuation, phase preserved. With speech protection the
 * frequency estimate is frozen during speech.
 */
public final class SpectralHumRemoval implements SpectralModifier {

    private final AdaptiveHumRemovalSettings settings;
    private final SpeechGate speechGate;
    private double estimatedF0;
    private boolean initialized;

    public SpectralHumRemoval(AdaptiveHumRemovalSettings settings, SpeechGate speechGate) {
        this.settings = settings;
        this.speechGate = speechGate == null ? SpeechGate.NEVER : speechGate;
    }

    public void modify(double[] real, double[] imag, int sampleRateHz, int frameStartSample) {
        int n = real.length;
        int half = n / 2;
        double binHz = (double) sampleRateHz / n;
        if (!initialized) {
            estimatedF0 = settings.getBaseFrequencyHz();
            initialized = true;
        }
        boolean speech = settings.isSpeechProtection()
                && speechGate.isSpeech(frameStartSample + n / 2);
        if (!speech) {
            double low = settings.getBaseFrequencyHz() - settings.getSearchRangeHz();
            double high = settings.getBaseFrequencyHz() + settings.getSearchRangeHz();
            int loBin = Spectra.binOf(low, binHz, half);
            int hiBin = Spectra.binOf(high, binHz, half);
            int peakBin = loBin;
            double peak = -1.0d;
            for (int k = loBin; k <= hiBin; k++) {
                double m = Spectra.magnitude(real, imag, k);
                if (m > peak) {
                    peak = m;
                    peakBin = k;
                }
            }
            double peakFreq = peakBin * binHz;
            estimatedF0 += settings.getAdaptationSpeed() * (peakFreq - estimatedF0);
            estimatedF0 = clampF0(estimatedF0);
        }
        double g = Math.pow(10.0d, -settings.getMaxAttenuationDb() / 20.0d);
        double nyquist = sampleRateHz / 2.0d;
        for (int h = 1; h <= settings.getHarmonics(); h++) {
            double freq = estimatedF0 * h;
            if (freq <= 0.0d || freq >= nyquist) {
                continue;
            }
            int center = Spectra.binOf(freq, binHz, half);
            for (int k = center - 1; k <= center + 1; k++) {
                if (k >= 1 && k <= half) {
                    Spectra.applyGain(real, imag, k, g);
                }
            }
        }
    }

    private double clampF0(double value) {
        double low = settings.getBaseFrequencyHz() - settings.getSearchRangeHz();
        double high = settings.getBaseFrequencyHz() + settings.getSearchRangeHz();
        if (Double.isNaN(value)) {
            return settings.getBaseFrequencyHz();
        }
        return value < low ? low : (value > high ? high : value);
    }
}
