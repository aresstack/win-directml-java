package com.aresstack.windirectml.inference.whisper.audio;

/**
 * Generate the Slaney-scaled, area-normalized triangular filters used by
 * librosa.filters.mel(sr=16000, n_fft=400, htk=false, norm="slaney").
 */
public final class WhisperMelFilterBank {
    private static final double LINEAR_FREQUENCY_SPACING = 200.0 / 3.0;
    private static final double LOG_FREQUENCY_START = 1000.0;
    private static final double LOG_MEL_START = LOG_FREQUENCY_START / LINEAR_FREQUENCY_SPACING;
    private static final double LOG_STEP = Math.log(6.4) / 27.0;

    private WhisperMelFilterBank() {
    }

    public static float[][] create(int count) {
        if (count != 80 && count != 128) {
            throw new IllegalArgumentException("Whisper supports 80 or 128 Mel filters: " + count);
        }
        double maxMel = toMel(8000.0);
        double[] frequencies = new double[count + 2];
        for (int i = 0; i < frequencies.length; i++) {
            frequencies[i] = toHertz(maxMel * i / (count + 1));
        }
        float[][] filters = new float[count][201];
        for (int band = 0; band < count; band++) {
            double lower = frequencies[band];
            double center = frequencies[band + 1];
            double upper = frequencies[band + 2];
            double normalization = 2.0 / (upper - lower);
            for (int bin = 0; bin <= 200; bin++) {
                double frequency = bin * 40.0;
                double lowerSlope = (frequency - lower) / (center - lower);
                double upperSlope = (upper - frequency) / (upper - center);
                filters[band][bin] = (float) (Math.max(0.0, Math.min(lowerSlope, upperSlope)) * normalization);
            }
        }
        return filters;
    }

    private static double toMel(double frequency) {
        if (frequency < LOG_FREQUENCY_START) {
            return frequency / LINEAR_FREQUENCY_SPACING;
        }
        return LOG_MEL_START + Math.log(frequency / LOG_FREQUENCY_START) / LOG_STEP;
    }

    private static double toHertz(double mel) {
        if (mel < LOG_MEL_START) {
            return mel * LINEAR_FREQUENCY_SPACING;
        }
        return LOG_FREQUENCY_START * Math.exp(LOG_STEP * (mel - LOG_MEL_START));
    }
}
