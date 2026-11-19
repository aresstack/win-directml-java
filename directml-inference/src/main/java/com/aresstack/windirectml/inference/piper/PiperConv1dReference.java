package com.aresstack.windirectml.inference.piper;

/** Compute the CPU reference for Piper's grouped one-dimensional cross-correlation. */
public final class PiperConv1dReference {

    private PiperConv1dReference() {
    }

    public static float[][] apply(float[][] input, float[][][] weight, float[] bias,
                                  int stride, int dilation, int padLeft, int padRight, int groups) {
        validateParameters(stride, dilation, padLeft, padRight, groups);
        int channels = input.length;
        int outputChannels = weight.length;
        if (channels == 0 || outputChannels == 0 || channels % groups != 0
                || outputChannels % groups != 0 || input[0].length == 0) {
            throw new IllegalArgumentException("Invalid grouped convolution channels");
        }
        int inputWidth = input[0].length;
        int channelsPerGroup = channels / groups;
        if (weight[0].length == 0) {
            throw new IllegalArgumentException("Convolution weights must have input channels");
        }
        int kernelWidth = weight[0][0].length;
        for (float[] channel : input) {
            if (channel.length != inputWidth) {
                throw new IllegalArgumentException("Inconsistent input channel length");
            }
        }
        for (float[][] outputChannel : weight) {
            if (outputChannel.length != channelsPerGroup) {
                throw new IllegalArgumentException("Incorrect convolution group size");
            }
            for (float[] kernel : outputChannel) {
                if (kernel.length != kernelWidth) {
                    throw new IllegalArgumentException("Inconsistent convolution kernel width");
                }
            }
        }
        if (kernelWidth == 0 || (bias != null && bias.length != outputChannels)) {
            throw new IllegalArgumentException("Invalid convolution weights or bias");
        }
        long numerator = (long) inputWidth + padLeft + padRight
                - (long) dilation * (kernelWidth - 1) - 1;
        if (numerator < 0 || numerator / stride + 1 > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Convolution output has no valid samples");
        }
        int outputWidth = (int) (numerator / stride + 1);
        float[][] result = new float[outputChannels][outputWidth];
        int outputsPerGroup = outputChannels / groups;
        for (int outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
            int inputStart = (outputChannel / outputsPerGroup) * channelsPerGroup;
            for (int sample = 0; sample < outputWidth; sample++) {
                float value = bias == null ? 0f : bias[outputChannel];
                for (int inputChannel = 0; inputChannel < channelsPerGroup; inputChannel++) {
                    for (int kernel = 0; kernel < kernelWidth; kernel++) {
                        int position = sample * stride - padLeft + kernel * dilation;
                        if (position >= 0 && position < inputWidth) {
                            value += input[inputStart + inputChannel][position]
                                    * weight[outputChannel][inputChannel][kernel];
                        }
                    }
                }
                result[outputChannel][sample] = value;
            }
        }
        return result;
    }

    static void validateParameters(int stride, int dilation, int padLeft, int padRight, int groups) {
        if (stride < 1 || dilation < 1 || padLeft < 0 || padRight < 0 || groups < 1) {
            throw new IllegalArgumentException("Invalid Piper convolution parameters");
        }
    }
}
