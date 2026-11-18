package com.aresstack.windirectml.inference.whisper;

import java.io.IOException;

/** Implement Whisper's float32 math independently of model storage and audio I/O. */
final class WhisperMath {
    private WhisperMath() {}

    static float[] project(float[] input, WhisperCheckpoint.Tensor weights,
                           WhisperCheckpoint.Tensor bias) {
        int outputs = weights.dimension(0);
        int inputs = weights.dimension(1);
        if (input.length != inputs) throw new IllegalArgumentException("Projection width mismatch");
        float[] w = weights.values();
        float[] b = bias == null ? null : bias.values();
        float[] result = new float[outputs];
        for (int row = 0; row < outputs; row++) {
            int offset = row * inputs;
            double sum = b == null ? 0 : b[row];
            for (int j = 0; j < inputs; j++) sum += w[offset + j] * input[j];
            result[row] = (float) sum;
        }
        return result;
    }

    static float[][] project(float[][] input, WhisperCheckpoint.Tensor w, WhisperCheckpoint.Tensor b) {
        float[][] result = new float[input.length][];
        for (int i = 0; i < input.length; i++) result[i] = project(input[i], w, b);
        return result;
    }

    static float[] norm(float[] input, WhisperCheckpoint.Tensor weight, WhisperCheckpoint.Tensor bias) {
        float[] w = weight.values();
        float[] b = bias.values();
        double mean = 0;
        for (float value : input) mean += value;
        mean /= input.length;
        double variance = 0;
        for (float value : input) variance += (value - mean) * (value - mean);
        double factor = 1.0 / Math.sqrt(variance / input.length + 1e-5);
        float[] result = new float[input.length];
        for (int i = 0; i < input.length; i++)
            result[i] = (float) ((input[i] - mean) * factor * w[i] + b[i]);
        return result;
    }

    static float[][] norm(float[][] input, WhisperCheckpoint.Tensor w, WhisperCheckpoint.Tensor b) {
        float[][] result = new float[input.length][];
        for (int i = 0; i < input.length; i++) result[i] = norm(input[i], w, b);
        return result;
    }

    static float[] add(float[] left, float[] right) {
        float[] result = new float[left.length];
        for (int i = 0; i < result.length; i++) result[i] = left[i] + right[i];
        return result;
    }

    static float[][] add(float[][] left, float[][] right) {
        float[][] result = new float[left.length][];
        for (int i = 0; i < result.length; i++) result[i] = add(left[i], right[i]);
        return result;
    }

    static float gelu(float x) {
        double z = Math.abs(x) / Math.sqrt(2);
        double t = 1 / (1 + 0.3275911 * z);
        double erf = 1 - (((((1.061405429 * t - 1.453152027) * t
                + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t)
                * Math.exp(-z * z);
        return (float) (0.5 * x * (1 + Math.copySign(erf, x)));
    }

    static float[] feedForward(float[] input, WhisperCheckpoint.Tensor first,
                               WhisperCheckpoint.Tensor firstBias, WhisperCheckpoint.Tensor second,
                               WhisperCheckpoint.Tensor secondBias) {
        float[] hidden = project(input, first, firstBias);
        for (int i = 0; i < hidden.length; i++) hidden[i] = gelu(hidden[i]);
        return project(hidden, second, secondBias);
    }

    static float[][] feedForward(float[][] input, WhisperCheckpoint.Tensor first,
                                 WhisperCheckpoint.Tensor firstBias, WhisperCheckpoint.Tensor second,
                                 WhisperCheckpoint.Tensor secondBias) {
        float[][] result = new float[input.length][];
        for (int i = 0; i < input.length; i++)
            result[i] = feedForward(input[i], first, firstBias, second, secondBias);
        return result;
    }

    static float[][] convolution(float[][] input, WhisperCheckpoint.Tensor weights,
                                  WhisperCheckpoint.Tensor bias, int stride) {
        int channels = weights.dimension(0);
        int inputs = weights.dimension(1);
        weights.requireShape(channels, inputs, 3);
        if (input.length == 0 || input[0].length != inputs) throw new IllegalArgumentException("Convolution channels mismatch");
        float[] kernel = weights.values();
        float[] b = bias.values();
        int count = (input.length - 1) / stride + 1;
        float[][] result = new float[count][channels];
        for (int step = 0; step < count; step++) {
            int center = step * stride;
            for (int channel = 0; channel < channels; channel++) {
                int kernelStart = channel * inputs * 3;
                double sum = b[channel];
                for (int in = 0; in < inputs; in++) {
                    int at = kernelStart + in * 3;
                    if (center > 0) sum += input[center - 1][in] * kernel[at];
                    sum += input[center][in] * kernel[at + 1];
                    if (center + 1 < input.length) sum += input[center + 1][in] * kernel[at + 2];
                }
                result[step][channel] = gelu((float) sum);
            }
        }
        return result;
    }

    static float[] attention(float[] query, float[][] keys, float[][] values, int heads, int length) {
        int width = query.length;
        int perHead = width / heads;
        double factor = 1 / Math.sqrt(perHead);
        float[] result = new float[width];
        float[] scores = new float[length];
        for (int head = 0; head < heads; head++) {
            int base = head * perHead;
            double max = -Double.MAX_VALUE;
            for (int t = 0; t < length; t++) {
                double dot = 0;
                for (int c = 0; c < perHead; c++) dot += query[base + c] * keys[t][base + c];
                double score = dot * factor;
                scores[t] = (float) score;
                max = Math.max(max, score);
            }
            double normalizer = 0;
            for (int t = 0; t < length; t++) {
                scores[t] = (float) Math.exp(scores[t] - max);
                normalizer += scores[t];
            }
            for (int t = 0; t < length; t++) {
                float probability = (float) (scores[t] / normalizer);
                for (int c = 0; c < perHead; c++)
                    result[base + c] += probability * values[t][base + c];
            }
        }
        return result;
    }

    static float[][] attention(float[][] queries, float[][] keys, float[][] values, int heads) {
        float[][] result = new float[queries.length][];
        for (int i = 0; i < queries.length; i++)
            result[i] = attention(queries[i], keys, values, heads, keys.length);
        return result;
    }

    static WhisperCheckpoint.Tensor w(WhisperCheckpoint model, String prefix) throws IOException {
        return model.require(prefix + ".weight");
    }

    static WhisperCheckpoint.Tensor b(WhisperCheckpoint model, String prefix) throws IOException {
        return model.require(prefix + ".bias");
    }
}
