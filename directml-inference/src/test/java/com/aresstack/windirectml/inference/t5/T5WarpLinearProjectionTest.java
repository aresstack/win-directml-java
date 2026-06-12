package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * T5's WARP linear projection is now a thin adapter over the shared {@code WarpDenseProjection}. These tests pin the
 * preserved T5 contract: rank-2 validation (device-free) and matvec/sequence results on WARP.
 */
class T5WarpLinearProjectionTest {

    @Test
    void fromRejectsNonRank2Weights() {
        // Rank is checked before native bindings are required → runs in CI without a device.
        T5TensorData rank1 = T5TensorData.reference("w", new long[]{6}, new float[]{0, 1, 2, 3, 4, 5});
        assertThrows(IllegalArgumentException.class, () -> T5WarpLinearProjection.from(null, rank1));
    }

    @Test
    void warpApplyMatchesReferenceMatVec() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML");
        int outputSize = 2;
        int inputSize = 3;
        float[] weights = {
                1.0f, 2.0f, 3.0f,
                -1.0f, 0.5f, 4.0f
        };
        float[] input = {2.0f, -1.0f, 0.25f};
        float[] reference = referenceMatVec(weights, outputSize, inputSize, input);
        T5TensorData weight = T5TensorData.reference("w", new long[]{outputSize, inputSize}, weights);

        try (WindowsBindings windowsBindings = new WindowsBindings()) {
            windowsBindings.init("warp");
            try (T5WarpLinearProjection projection = T5WarpLinearProjection.from(windowsBindings, weight)) {
                assertEquals(inputSize, projection.inputSize());
                assertEquals(outputSize, projection.outputSize());
                assertArrayEquals(reference, projection.apply(input), 0.001f);

                float[] second = {0.5f, 1.0f, -2.0f};
                float[] secondReference = referenceMatVec(weights, outputSize, inputSize, second);
                float[] sequence = new float[2 * inputSize];
                System.arraycopy(input, 0, sequence, 0, inputSize);
                System.arraycopy(second, 0, sequence, inputSize, inputSize);
                float[] sequenceOut = projection.applySequence(sequence, 2, inputSize);
                assertArrayEquals(reference, java.util.Arrays.copyOfRange(sequenceOut, 0, outputSize), 0.001f);
                assertArrayEquals(secondReference,
                        java.util.Arrays.copyOfRange(sequenceOut, outputSize, 2 * outputSize), 0.001f);
            }
        }
    }

    @Test
    void fp32RuntimeTensorTakesByteBufferPathAndMatchesFloatArrayPath() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML");
        int outputSize = 2;
        int inputSize = 3;
        float[] weights = {
                1.0f, 2.0f, 3.0f,
                -1.0f, 0.5f, 4.0f
        };
        float[] input = {2.0f, -1.0f, 0.25f};

        // FP32 runtime-tensor weight -> heap-light ByteBuffer upload path.
        java.nio.ByteBuffer raw = java.nio.ByteBuffer.allocate(weights.length * Float.BYTES)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (float w : weights) {
            raw.putFloat(w);
        }
        raw.flip();
        com.aresstack.windirectml.inference.model.RuntimeTensor tensor =
                new com.aresstack.windirectml.inference.model.RuntimeTensor("w", new long[]{outputSize, inputSize},
                        com.aresstack.windirectml.windows.OnnxModelReader.ONNX_FLOAT, raw, weights.length * Float.BYTES);
        T5TensorData byteBufferWeight = T5TensorData.from(tensor);
        assertNotNull(byteBufferWeight.fp32LittleEndianSource(), "FP32 tensor must take the ByteBuffer path");

        // Reference (float[]) weight with identical values -> float[] fallback path.
        T5TensorData floatArrayWeight = T5TensorData.reference("w", new long[]{outputSize, inputSize}, weights);

        try (WindowsBindings windowsBindings = new WindowsBindings()) {
            windowsBindings.init("warp");
            try (T5WarpLinearProjection viaBytes = T5WarpLinearProjection.from(windowsBindings, byteBufferWeight);
                 T5WarpLinearProjection viaFloats = T5WarpLinearProjection.from(windowsBindings, floatArrayWeight)) {
                float[] expected = viaFloats.apply(input);
                float[] actual = viaBytes.apply(input);
                assertArrayEquals(expected, actual, 0.0f,
                        "T5 ByteBuffer upload path must be byte-identical to the float[] path");
            }
        }
    }

    private static float[] referenceMatVec(float[] weights, int outputSize, int inputSize, float[] input) {
        float[] output = new float[outputSize];
        for (int o = 0; o < outputSize; o++) {
            float sum = 0.0f;
            for (int i = 0; i < inputSize; i++) {
                sum += weights[o * inputSize + i] * input[i];
            }
            output[o] = sum;
        }
        return output;
    }
}
