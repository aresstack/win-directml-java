package com.aresstack.windirectml.inference.whisper;

import com.aresstack.windirectml.inference.simd.FloatMathOps;
import com.aresstack.windirectml.inference.simd.ScalarFloatMathOps;
import com.aresstack.windirectml.inference.simd.SimdMath;

import java.io.IOException;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

/**
 * Implement Whisper's float32 math independently of model storage and audio I/O.
 * Batched operations process independent rows in parallel; every row is computed
 * exactly as in sequential order, so results do not depend on the thread count.
 *
 * <p>Dot products and scaled additions use {@link SimdMath#provider()}: SIMD through the
 * Java Vector API when {@code jdk.incubator.vector} is present, otherwise a scalar
 * fallback. {@code -Dwindirectml.whisper.simd=false} forces the scalar path.
 */
final class WhisperMath {
    private static final int PARALLEL_ROWS = 16;
    static final FloatMathOps MATH = "false".equalsIgnoreCase(System.getProperty("windirectml.whisper.simd"))
            ? new ScalarFloatMathOps() : SimdMath.provider();

    private WhisperMath() {}

    static void forEachRow(int rows, IntConsumer action) {
        if (rows < PARALLEL_ROWS) {
            for (int row = 0; row < rows; row++) action.accept(row);
        } else {
            IntStream.range(0, rows).parallel().forEach(action);
        }
    }

    static float[] project(float[] input, WhisperCheckpoint.Tensor weights,
                           WhisperCheckpoint.Tensor bias) {
        int outputs = weights.dimension(0);
        int inputs = weights.dimension(1);
        if (input.length != inputs) throw new IllegalArgumentException("Projection width mismatch");
        float[] w = weights.values();
        float[] b = bias == null ? null : bias.values();
        float[] result = new float[outputs];
        for (int row = 0; row < outputs; row++) {
            float dot = MATH.dot(w, row * inputs, input, 0, inputs);
            result[row] = b == null ? dot : dot + b[row];
        }
        return result;
    }

    static float[][] project(float[][] input, WhisperCheckpoint.Tensor w, WhisperCheckpoint.Tensor b) {
        float[][] result = new float[input.length][];
        forEachRow(input.length, i -> result[i] = project(input[i], w, b));
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
        forEachRow(input.length, i -> result[i] = norm(input[i], w, b));
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
        forEachRow(input.length, i -> result[i] = feedForward(input[i], first, firstBias, second, secondBias));
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
        // Reorder [channel][input][tap] to [channel][tap][input] so every tap is one contiguous dot product.
        float[] taps = new float[kernel.length];
        for (int channel = 0; channel < channels; channel++) {
            for (int in = 0; in < inputs; in++) {
                for (int tap = 0; tap < 3; tap++) {
                    taps[(channel * 3 + tap) * inputs + in] = kernel[(channel * inputs + in) * 3 + tap];
                }
            }
        }
        int count = (input.length - 1) / stride + 1;
        float[][] result = new float[count][channels];
        forEachRow(count, step -> {
            int center = step * stride;
            for (int channel = 0; channel < channels; channel++) {
                float sum = b[channel];
                for (int tap = 0; tap < 3; tap++) {
                    int frame = center + tap - 1;
                    if (frame >= 0 && frame < input.length) {
                        sum += MATH.dot(input[frame], 0, taps, (channel * 3 + tap) * inputs, inputs);
                    }
                }
                result[step][channel] = gelu(sum);
            }
        });
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
                double score = MATH.dot(query, base, keys[t], base, perHead) * factor;
                scores[t] = (float) score;
                max = Math.max(max, score);
            }
            double normalizer = 0;
            for (int t = 0; t < length; t++) {
                scores[t] = (float) Math.exp(scores[t] - max);
                normalizer += scores[t];
            }
            for (int t = 0; t < length; t++) {
                MATH.axpy(result, base, (float) (scores[t] / normalizer), values[t], base, perHead);
            }
        }
        return result;
    }

    static float[][] attention(float[][] queries, float[][] keys, float[][] values, int heads) {
        float[][] result = new float[queries.length][];
        forEachRow(queries.length, i -> result[i] = attention(queries[i], keys, values, heads, keys.length));
        return result;
    }

    static WhisperCheckpoint.Tensor w(WhisperCheckpoint model, String prefix) throws IOException {
        return model.require(prefix + ".weight");
    }

    static WhisperCheckpoint.Tensor b(WhisperCheckpoint model, String prefix) throws IOException {
        return model.require(prefix + ".bias");
    }
}
