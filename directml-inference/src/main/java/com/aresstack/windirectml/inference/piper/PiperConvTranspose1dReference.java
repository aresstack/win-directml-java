package com.aresstack.windirectml.inference.piper;

/** Compute a grouped transposed one-dimensional convolution as a VITS decoder baseline. */
public final class PiperConvTranspose1dReference {

    private PiperConvTranspose1dReference() {
    }

    public static float[][] apply(float[][] input, float[][][] weight, float[] bias,
                                  int stride, int dilation, int padLeft, int padRight,
                                  int groups, int outputPadding) {
        PiperConv1dReference.validateParameters(stride, dilation, padLeft, padRight, groups);
        if (outputPadding < 0 || outputPadding >= stride || input.length == 0
                || weight.length != input.length || input.length % groups != 0) {
            throw new IllegalArgumentException("Invalid transposed convolution configuration");
        }
        int inputWidth = input[0].length;
        int channelsPerGroup = input.length / groups;
        if (inputWidth == 0 || weight[0].length == 0 || weight[0][0].length == 0) {
            throw new IllegalArgumentException("Empty transposed convolution input");
        }
        int outputsPerGroup = weight[0].length;
        int kernelWidth = weight[0][0].length;
        int outputChannels = outputsPerGroup * groups;
        for (int channel = 0; channel < input.length; channel++) {
            if (input[channel].length != inputWidth || weight[channel].length != outputsPerGroup) {
                throw new IllegalArgumentException("Inconsistent transposed convolution channels");
            }
            for (float[] kernel : weight[channel]) {
                if (kernel.length != kernelWidth) {
                    throw new IllegalArgumentException("Inconsistent transposed convolution kernel");
                }
            }
        }
        if (bias != null && bias.length != outputChannels) {
            throw new IllegalArgumentException("Incorrect transposed convolution bias");
        }
        int outputWidth = (inputWidth - 1) * stride - padLeft - padRight
                + dilation * (kernelWidth - 1) + outputPadding + 1;
        if (outputWidth < 1) {
            throw new IllegalArgumentException("Transposed convolution output has no samples");
        }
        float[][] output = new float[outputChannels][outputWidth];
        for (int inputChannel = 0; inputChannel < input.length; inputChannel++) {
            int outputStart = (inputChannel / channelsPerGroup) * outputsPerGroup;
            for (int sample = 0; sample < inputWidth; sample++) {
                for (int outputChannel = 0; outputChannel < outputsPerGroup; outputChannel++) {
                    for (int kernel = 0; kernel < kernelWidth; kernel++) {
                        int target = sample * stride - padLeft + kernel * dilation;
                        if (target >= 0 && target < outputWidth) {
                            output[outputStart + outputChannel][target]
                                    += input[inputChannel][sample]
                                    * weight[inputChannel][outputChannel][kernel];
                        }
                    }
                }
            }
        }
        if (bias != null) {
            for (int channel = 0; channel < outputChannels; channel++) {
                for (int position = 0; position < outputWidth; position++) {
                    output[channel][position] += bias[channel];
                }
            }
        }
        return output;
    }
}
