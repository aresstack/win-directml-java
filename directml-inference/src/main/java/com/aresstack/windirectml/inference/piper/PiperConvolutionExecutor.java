package com.aresstack.windirectml.inference.piper;

import com.aresstack.windirectml.windows.OnnxModelReader.OnnxNode;

import java.util.List;

/**
 * Adapt ONNX NCL convolution tensors to the independent 1-D CPU reference kernels.
 * Keep the graph interpreter unaware of channel layouts and decoder upsampling.
 */
public final class PiperConvolutionExecutor {

    private PiperConvolutionExecutor() {
    }

    public static PiperTensor apply(OnnxNode node, PiperTensor source,
                                    PiperTensor filters, PiperTensor bias) {
        int[] inputShape = source.shape();
        int[] weightShape = filters.shape();
        if (inputShape.length != 3 || weightShape.length != 3
                || inputShape[1] != ("ConvTranspose".equals(node.opType())
                        ? weightShape[0] : weightShape[1] * Math.toIntExact(node.getInt("group", 1)))) {
            throw new IllegalArgumentException("Piper 1-D NCL convolution tensor shapes are incompatible");
        }

        int batches = inputShape[0];
        int inputChannels = inputShape[1];
        int inputWidth = inputShape[2];
        int outputs = "ConvTranspose".equals(node.opType())
                ? weightShape[1] * Math.toIntExact(node.getInt("group", 1))
                : weightShape[0];
        int channelsPerFilter = weightShape[1];
        int kernelSize = weightShape[2];
        int groups = Math.toIntExact(node.getInt("group", 1));
        int stride = attribute(node, "strides", 1);
        int dilation = attribute(node, "dilations", 1);
        List<Long> pads = node.getInts("pads");
        if (!pads.isEmpty() && pads.size() != 2) {
            throw new IllegalArgumentException("Piper convolution expects 1-D padding");
        }
        int left = pads.isEmpty() ? 0 : Math.toIntExact(pads.get(0));
        int right = pads.isEmpty() ? 0 : Math.toIntExact(pads.get(1));
        int outputPadding = attribute(node, "output_padding", 0);

        float[][][] weights = unpackFilters(filters.requireFloat(),
                weightShape[0], channelsPerFilter, kernelSize);
        float[] values = source.requireFloat();
        float[] biases = bias == null ? null : bias.requireFloat();
        float[] flattened = null;
        int outputWidth = 0;
        for (int batch = 0; batch < batches; batch++) {
            float[][] channels = unpackChannels(values, batch, inputChannels, inputWidth);
            float[][] result = "ConvTranspose".equals(node.opType())
                    ? PiperConvTranspose1dReference.apply(channels, weights, biases,
                            stride, dilation, left, right, groups, outputPadding)
                    : PiperConv1dReference.apply(channels, weights, biases,
                            stride, dilation, left, right, groups);
            if (flattened == null) {
                outputWidth = result[0].length;
                flattened = new float[Math.multiplyExact(
                        Math.multiplyExact(batches, outputs), outputWidth)];
            }
            for (int channel = 0; channel < outputs; channel++) {
                System.arraycopy(result[channel], 0, flattened,
                        (batch * outputs + channel) * outputWidth, outputWidth);
            }
        }
        if (flattened == null) {
            throw new IllegalArgumentException("Piper convolution requires at least one batch");
        }
        return PiperTensor.floats(flattened, batches, outputs, outputWidth);
    }

    private static float[][] unpackChannels(float[] values, int batch, int channels, int width) {
        float[][] result = new float[channels][width];
        for (int channel = 0; channel < channels; channel++) {
            System.arraycopy(values, (batch * channels + channel) * width,
                    result[channel], 0, width);
        }
        return result;
    }

    private static float[][][] unpackFilters(float[] values,
                                             int channels, int perGroup, int kernelSize) {
        float[][][] result = new float[channels][perGroup][kernelSize];
        for (int channel = 0; channel < channels; channel++) {
            for (int group = 0; group < perGroup; group++) {
                System.arraycopy(values, (channel * perGroup + group) * kernelSize,
                        result[channel][group], 0, kernelSize);
            }
        }
        return result;
    }

    private static int attribute(OnnxNode node, String name, int fallback) {
        List<Long> parameters = node.getInts(name);
        return parameters.isEmpty() ? fallback : Math.toIntExact(parameters.get(0));
    }
}
