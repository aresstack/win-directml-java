package com.aresstack.windirectml.inference.piper;

import com.aresstack.windirectml.windows.OnnxModelReader.OnnxNode;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

final class PiperConvolutionExecutorTest {

    @Test
    void unpacksAndExecutesForwardOnnxNclConvolution() {
        OnnxNode node = node("Conv", Map.of(
                "strides", List.of(1L), "pads", List.of(1L, 0L)));
        PiperTensor signal = PiperTensor.floats(new float[]{1, 2, 3}, 1, 1, 3);
        PiperTensor filters = PiperTensor.floats(new float[]{2, 1}, 1, 1, 2);
        PiperTensor biases = PiperTensor.floats(new float[]{1}, 1);

        PiperTensor output = PiperConvolutionExecutor.apply(node, signal, filters, biases);

        assertArrayEquals(new int[]{1, 1, 3}, output.shape());
        assertArrayEquals(new float[]{2, 5, 8}, output.floatData(), 0f);
    }

    @Test
    void upscalesDecoderTensorWithTransposedConvolutionAndBias() {
        OnnxNode node = node("ConvTranspose", Map.of(
                "strides", List.of(2L), "pads", List.of(0L, 0L),
                "kernel_shape", List.of(2L)));
        PiperTensor signal = PiperTensor.floats(new float[]{1, 2}, 1, 1, 2);
        PiperTensor filters = PiperTensor.floats(new float[]{1, 2}, 1, 1, 2);
        PiperTensor biases = PiperTensor.floats(new float[]{0.5f}, 1);

        PiperTensor output = PiperConvolutionExecutor.apply(node, signal, filters, biases);

        assertArrayEquals(new int[]{1, 1, 4}, output.shape());
        assertArrayEquals(new float[]{1.5f, 2.5f, 2.5f, 4.5f},
                output.floatData(), 0f);
    }

    @Test
    void preservesBatchSeparationInUpsampling() {
        OnnxNode node = node("ConvTranspose", Map.of("strides", List.of(2L)));
        PiperTensor input = PiperTensor.floats(new float[]{1, 2, 10, 20}, 2, 1, 2);
        PiperTensor filter = PiperTensor.floats(new float[]{2}, 1, 1, 1);

        PiperTensor output = PiperConvolutionExecutor.apply(node, input, filter, null);

        assertArrayEquals(new int[]{2, 1, 3}, output.shape());
        assertArrayEquals(new float[]{2, 0, 4, 20, 0, 40}, output.floatData(), 0f);
    }

    @Test
    void rejectsIncorrectChannelLayouts() {
        OnnxNode node = node("ConvTranspose", Map.of());
        PiperTensor input = PiperTensor.floats(new float[]{1, 2}, 1, 1, 2);
        PiperTensor filters = PiperTensor.floats(new float[]{1, 2, 3, 4}, 2, 1, 2);

        assertThrows(IllegalArgumentException.class,
                () -> PiperConvolutionExecutor.apply(node, input, filters, null));
    }

    private static OnnxNode node(String operator, Map<String, Object> attrs) {
        return new OnnxNode(operator, List.of("x", "weights", "bias"), List.of("y"), attrs);
    }
}
