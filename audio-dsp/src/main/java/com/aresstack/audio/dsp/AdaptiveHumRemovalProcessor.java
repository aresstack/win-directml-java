package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Remove mains hum whose frequency drifts slightly. Working block by block, a single-bin {@link Goertzel}
 * scan over the search range around the expected base frequency estimates the current fundamental; the
 * estimate is smoothed with the adaptation speed and used to retune narrow band-pass filters at the
 * fundamental and its harmonics. The isolated hum is then subtracted, limited by the maximum attenuation,
 * so the rest of the signal is largely preserved. With speech protection and an upstream track, the
 * frequency estimate is frozen during speech (whose energy would otherwise bias it); the notches still run.
 *
 * <p>No FFT/STFT and no new dependency — only the shared biquad and a one-frequency Goertzel probe. A fresh
 * instance per run keeps state isolated.</p>
 */
public final class AdaptiveHumRemovalProcessor {

    private static final double BANDWIDTH_HZ = 4.0d;   // narrow notch bandwidth per harmonic
    private static final int BLOCK_MILLIS = 250;
    private static final double SCAN_STEP_HZ = 0.25d;

    private final AdaptiveHumRemovalSettings settings;

    public AdaptiveHumRemovalProcessor(AdaptiveHumRemovalSettings settings) {
        if (settings == null) {
            throw new IllegalArgumentException("Adaptive hum-removal settings must not be null.");
        }
        this.settings = settings;
    }

    public void process(short[] samples, int count, PcmAudioFormat format, SpeechActivityTrack track) {
        int channels = Math.max(1, format.getChannels());
        int rate = format.getSampleRateHz();
        int frames = count / channels;
        if (frames <= 0 || rate <= 0) {
            return;
        }
        double[] mono = toMono(samples, frames, channels);
        double depthFactor = 1.0d - Math.pow(10.0d, -settings.getMaxAttenuationDb() / 20.0d);
        int harmonics = settings.getHarmonics();
        BiquadFilterState[][] states = new BiquadFilterState[harmonics][channels];
        for (int h = 0; h < harmonics; h++) {
            for (int c = 0; c < channels; c++) {
                states[h][c] = new BiquadFilterState();
            }
        }
        BiquadCoefficients[] coefficients = new BiquadCoefficients[harmonics];
        double estimatedF0 = settings.getBaseFrequencyHz();
        int blockN = Math.max(1, (int) Math.round(rate * BLOCK_MILLIS / 1000.0d));

        for (int start = 0; start < frames; start += blockN) {
            int blockLength = Math.min(blockN, frames - start);
            boolean speechBlock = settings.isSpeechProtection() && track != null
                    && isSpeech(track, start * channels);
            if (!speechBlock) {
                double best = scanFundamental(mono, start, blockLength, rate);
                estimatedF0 += settings.getAdaptationSpeed() * (best - estimatedF0);
                estimatedF0 = clampF0(estimatedF0);
            }
            retune(coefficients, rate, estimatedF0);
            int end = start + blockLength;
            for (int f = start; f < end; f++) {
                int base = f * channels;
                for (int c = 0; c < channels; c++) {
                    double x = samples[base + c];
                    double hum = 0.0d;
                    for (int h = 0; h < harmonics; h++) {
                        if (coefficients[h] != null) {
                            hum += states[h][c].process(coefficients[h], x);
                        }
                    }
                    samples[base + c] = clamp(x - depthFactor * hum);
                }
            }
        }
    }

    private void retune(BiquadCoefficients[] coefficients, int rate, double estimatedF0) {
        double nyquist = rate / 2.0d;
        for (int h = 0; h < coefficients.length; h++) {
            double freq = estimatedF0 * (h + 1);
            if (freq > 0.0d && freq < nyquist) {
                try {
                    coefficients[h] = BiquadCoefficients.bandPass(rate, freq, Math.max(0.5d, freq / BANDWIDTH_HZ));
                } catch (RuntimeException invalid) {
                    coefficients[h] = null;
                }
            } else {
                coefficients[h] = null;
            }
        }
    }

    private double scanFundamental(double[] mono, int start, int length, int rate) {
        double low = settings.getBaseFrequencyHz() - settings.getSearchRangeHz();
        double high = settings.getBaseFrequencyHz() + settings.getSearchRangeHz();
        double best = settings.getBaseFrequencyHz();
        double bestPower = -1.0d;
        for (double freq = Math.max(1.0d, low); freq <= high; freq += SCAN_STEP_HZ) {
            double power = Goertzel.power(mono, start, length, rate, freq);
            if (power > bestPower) {
                bestPower = power;
                best = freq;
            }
        }
        return best;
    }

    private double clampF0(double value) {
        double low = settings.getBaseFrequencyHz() - settings.getSearchRangeHz();
        double high = settings.getBaseFrequencyHz() + settings.getSearchRangeHz();
        if (Double.isNaN(value)) {
            return settings.getBaseFrequencyHz();
        }
        return value < low ? low : (value > high ? high : value);
    }

    private static boolean isSpeech(SpeechActivityTrack track, int interleavedIndex) {
        SpeechActivityMetadata frame = track.frameForInterleavedIndex(interleavedIndex);
        return frame != null && frame.isSpeechActive();
    }

    private static double[] toMono(short[] samples, int frames, int channels) {
        double[] mono = new double[frames];
        for (int f = 0; f < frames; f++) {
            int base = f * channels;
            double sum = 0.0d;
            for (int c = 0; c < channels; c++) {
                sum += samples[base + c];
            }
            mono[f] = sum / channels;
        }
        return mono;
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
