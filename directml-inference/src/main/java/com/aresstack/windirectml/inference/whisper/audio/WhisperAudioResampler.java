package com.aresstack.windirectml.inference.whisper.audio;

/**
 * Resample float PCM using a windowed, low-pass filtered sinc interpolation.
 * Preserve input resolution rather than quantizing through 16-bit PCM.
 */
public final class WhisperAudioResampler {
    private static final int FILTER_RADIUS = 32;
    private static final double PASSBAND_FRACTION = 0.95;

    private WhisperAudioResampler() {
    }

    public static float[] resample(float[] input, int sourceRate, int targetRate) {
        if (input == null || sourceRate <= 0 || targetRate <= 0) {
            throw new IllegalArgumentException("Samples and positive sample rates are required");
        }
        if (sourceRate == targetRate) {
            return input.clone();
        }
        if (input.length == 0) {
            return new float[0];
        }
        long targetLength = Math.round((double) input.length * targetRate / sourceRate);
        if (targetLength > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Resampled audio exceeds array size");
        }
        float[] output = new float[(int) targetLength];
        double downsampleFactor = Math.min(1.0, (double) targetRate / sourceRate);
        double cutoff = downsampleFactor * PASSBAND_FRACTION;
        int radius = (int) Math.ceil(FILTER_RADIUS / downsampleFactor);
        for (int sample = 0; sample < output.length; sample++) {
            double sourcePosition = (double) sample * sourceRate / targetRate;
            int nearest = (int) Math.floor(sourcePosition);
            double weighted = 0.0;
            double totalWeight = 0.0;
            for (int offset = -radius; offset <= radius; offset++) {
                int index = nearest + offset;
                if (index < 0 || index >= input.length) {
                    continue;
                }
                double distance = index - sourcePosition;
                double windowPosition = Math.abs(distance) / (radius + 1.0);
                double window = 0.5 + 0.5 * Math.cos(Math.PI * windowPosition);
                double weight = cutoff * sinc(cutoff * distance) * window;
                weighted += input[index] * weight;
                totalWeight += weight;
            }
            output[sample] = totalWeight == 0.0 ? 0.0f : (float) (weighted / totalWeight);
        }
        return output;
    }

    private static double sinc(double value) {
        if (Math.abs(value) < 1.0e-12) {
            return 1.0;
        }
        double angle = Math.PI * value;
        return Math.sin(angle) / angle;
    }
}
