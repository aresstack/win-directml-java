package com.aresstack.windirectml.inference.warp;

import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Shared, model-family-neutral WARP dense projection. Shape validation is device-free; the matvec/sequence behaviour
 * is verified on WARP (mirrors {@code DecoderOnlyWarpDenseProjectionTest}, demonstrating the neutral building block
 * behaves identically so decoder-only can later migrate onto it).
 */
class WarpDenseProjectionTest {

    @Test
    void rejectsInvalidShapesWithoutNeedingNativeBindings() {
        // Shape is validated before native bindings are required, so these run in CI without a device.
        assertThrows(IllegalArgumentException.class,
                () -> WarpDenseProjection.fromDequantizedWeights(null, "w", 0, 3, new float[0]));
        assertThrows(IllegalArgumentException.class,
                () -> WarpDenseProjection.fromDequantizedWeights(null, "w", 2, 0, new float[0]));
        assertThrows(IllegalArgumentException.class,
                () -> WarpDenseProjection.fromDequantizedWeights(null, "w", 2, 3, new float[5]));
    }

    @Test
    void byteBufferOverloadRejectsInvalidShapeAndEndiannessWithoutNativeBindings() {
        // Wrong remaining size (2x3 = 6 floats = 24 bytes expected).
        java.nio.ByteBuffer tooSmall = java.nio.ByteBuffer.allocate(20).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        assertThrows(IllegalArgumentException.class,
                () -> WarpDenseProjection.fromDequantizedWeights(null, "w", 2, 3, tooSmall));
        // Correct size but big-endian -> rejected as a contract violation.
        java.nio.ByteBuffer bigEndian = java.nio.ByteBuffer.allocate(24).order(java.nio.ByteOrder.BIG_ENDIAN);
        assertThrows(IllegalArgumentException.class,
                () -> WarpDenseProjection.fromDequantizedWeights(null, "w", 2, 3, bigEndian));
        // Degenerate shapes rejected before native bindings.
        java.nio.ByteBuffer any = java.nio.ByteBuffer.allocate(0).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        assertThrows(IllegalArgumentException.class,
                () -> WarpDenseProjection.fromDequantizedWeights(null, "w", 0, 3, any));
    }

    @Test
    void warpByteBufferUploadMatchesFloatArrayUpload() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML");
        int outputSize = 2;
        int inputSize = 3;
        float[] weights = {
                1.0f, 2.0f, 3.0f,
                -1.0f, 0.5f, 4.0f
        };
        float[] input = {2.0f, -1.0f, 0.25f};

        try (WindowsBindings windowsBindings = new WindowsBindings()) {
            windowsBindings.init("warp");
            // Direct, read-only LE buffer to mirror an mmap .wdmlpack slice.
            java.nio.ByteBuffer le = java.nio.ByteBuffer.allocateDirect(weights.length * Float.BYTES)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN);
            for (float w : weights) {
                le.putFloat(w);
            }
            le.flip();
            java.nio.ByteBuffer readOnly = le.asReadOnlyBuffer().order(java.nio.ByteOrder.LITTLE_ENDIAN);

            try (WarpDenseProjection viaFloats = WarpDenseProjection.fromDequantizedWeights(
                    windowsBindings, "floats", outputSize, inputSize, weights);
                 WarpDenseProjection viaBytes = WarpDenseProjection.fromDequantizedWeights(
                         windowsBindings, "bytes", outputSize, inputSize, readOnly)) {

                float[] expected = viaFloats.project(input);
                float[] actual = viaBytes.project(input);
                assertArrayEquals(expected, actual, 0.0f,
                        "ByteBuffer upload must produce byte-identical results to the float[] upload");

                // The source buffer's position/limit must be untouched by the upload.
                assertEquals(0, readOnly.position());
                assertEquals(weights.length * Float.BYTES, readOnly.limit());
            }
        }
    }

    @Test
    void warpProjectionMatchesReferenceMatVec() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML");
        int outputSize = 2;
        int inputSize = 3;
        float[] weights = {
                1.0f, 2.0f, 3.0f,
                -1.0f, 0.5f, 4.0f
        };
        float[] input = {2.0f, -1.0f, 0.25f};
        float[] reference = referenceMatVec(weights, outputSize, inputSize, input);

        try (WindowsBindings windowsBindings = new WindowsBindings()) {
            windowsBindings.init("warp");
            try (WarpDenseProjection projection = WarpDenseProjection.fromDequantizedWeights(
                    windowsBindings, "warp", outputSize, inputSize, weights)) {
                assertEquals(inputSize, projection.inputSize());
                assertEquals(outputSize, projection.outputSize());

                float[] viaAllocating = projection.project(input);
                assertArrayEquals(reference, viaAllocating, 0.001f);

                float[] into = new float[outputSize];
                projection.projectInto(input, into);
                assertArrayEquals(reference, into, 0.001f);

                // Sequence projection must equal the per-row projection of each row.
                float[] secondInput = {0.5f, 1.0f, -2.0f};
                float[] sequence = concat(input, secondInput);
                float[] secondReference = referenceMatVec(weights, outputSize, inputSize, secondInput);
                float[] sequenceOut = projection.projectSequence(sequence, 2);
                assertArrayEquals(reference, java.util.Arrays.copyOfRange(sequenceOut, 0, outputSize), 0.001f);
                assertArrayEquals(secondReference,
                        java.util.Arrays.copyOfRange(sequenceOut, outputSize, 2 * outputSize), 0.001f);
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

    private static float[] concat(float[] a, float[] b) {
        float[] result = new float[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }
}
