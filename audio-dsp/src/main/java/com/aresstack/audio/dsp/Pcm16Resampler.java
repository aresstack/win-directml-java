package com.aresstack.audio.dsp;

/**
 * Resample a mono 16-bit PCM buffer. Keep the original compatibility entry point with automatic
 * anti-aliasing and expose an explicit mode for profile-driven pipelines where the low-pass filter
 * is represented by its own visible block.
 */
public final class Pcm16Resampler {

    private Pcm16Resampler() {
    }

    /** Preserve the original behavior: filter before downsampling and use balanced interpolation. */
    public static short[] resample(short[] mono, int srcRate, int dstRate) {
        return resample(mono, srcRate, dstRate, ResamplingQuality.BALANCED, true);
    }

    /**
     * @param mono              mono 16-bit samples
     * @param srcRate           source sample rate in Hz
     * @param dstRate           destination sample rate in Hz
     * @param quality           interpolation quality
     * @param applyAntiAliasing apply the legacy hidden anti-alias filter; set false when a visible
     *                          low-pass block precedes this resampler
     */
    public static short[] resample(short[] mono, int srcRate, int dstRate,
                                   ResamplingQuality quality, boolean applyAntiAliasing) {
        if (mono == null) {
            throw new IllegalArgumentException("Samples must not be null.");
        }
        if (srcRate <= 0 || dstRate <= 0) {
            throw new IllegalArgumentException("Sample rates must be positive.");
        }
        int inputLength = mono.length;
        if (srcRate == dstRate || inputLength == 0) {
            return copy(mono);
        }
        short[] input = mono;
        if (applyAntiAliasing && dstRate < srcRate) {
            input = Pcm16LowPassFilter.filter(input, srcRate, 0.45d * dstRate);
        }
        long rounded = Math.round((double) inputLength * dstRate / srcRate);
        int outputLength = (int) Math.max(1L, rounded);
        short[] output = new short[outputLength];
        double step = (double) srcRate / dstRate;
        ResamplingQuality selected = quality == null ? ResamplingQuality.BALANCED : quality;
        for (int i = 0; i < outputLength; i++) {
            double sourcePosition = i * step;
            if (selected == ResamplingQuality.FAST) {
                output[i] = nearest(input, sourcePosition);
            } else if (selected == ResamplingQuality.BALANCED) {
                output[i] = linear(input, sourcePosition);
            } else {
                output[i] = cubic(input, sourcePosition);
            }
        }
        return output;
    }

    private static short nearest(short[] input, double sourcePosition) {
        int index = clampIndex((int) Math.round(sourcePosition), input.length);
        return input[index];
    }

    private static short linear(short[] input, double sourcePosition) {
        int index = (int) Math.floor(sourcePosition);
        double fraction = sourcePosition - index;
        int left = input[clampIndex(index, input.length)];
        int right = input[clampIndex(index + 1, input.length)];
        return clamp(left + (right - left) * fraction);
    }

    /** Apply Catmull-Rom interpolation for a smoother high-quality profile option. */
    private static short cubic(short[] input, double sourcePosition) {
        int index = (int) Math.floor(sourcePosition);
        double t = sourcePosition - index;
        double p0 = input[clampIndex(index - 1, input.length)];
        double p1 = input[clampIndex(index, input.length)];
        double p2 = input[clampIndex(index + 1, input.length)];
        double p3 = input[clampIndex(index + 2, input.length)];
        double t2 = t * t;
        double t3 = t2 * t;
        double value = 0.5d * ((2.0d * p1)
                + (-p0 + p2) * t
                + (2.0d * p0 - 5.0d * p1 + 4.0d * p2 - p3) * t2
                + (-p0 + 3.0d * p1 - 3.0d * p2 + p3) * t3);
        return clamp(value);
    }

    private static int clampIndex(int index, int length) {
        if (index < 0) {
            return 0;
        }
        if (index >= length) {
            return length - 1;
        }
        return index;
    }

    private static short[] copy(short[] input) {
        short[] copy = new short[input.length];
        System.arraycopy(input, 0, copy, 0, input.length);
        return copy;
    }

    private static short clamp(double value) {
        if (value > Short.MAX_VALUE) {
            return Short.MAX_VALUE;
        }
        if (value < Short.MIN_VALUE) {
            return Short.MIN_VALUE;
        }
        return (short) Math.round(value);
    }
}
