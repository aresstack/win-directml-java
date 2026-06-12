package com.aresstack.windirectml.inference.decoderonly;

import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class DecoderOnlyWarpDenseProjectionTest {

    @Test
    void rejectsInvalidShapesWithoutNeedingNativeBindings() {
        // Shape is validated by the shared WarpDenseProjection before native bindings are required → device-free.
        assertThrows(IllegalArgumentException.class,
                () -> DecoderOnlyWarpDenseProjection.fromRowMajorWeights(null, "w", 0, 3, new float[0]));
        assertThrows(IllegalArgumentException.class,
                () -> DecoderOnlyWarpDenseProjection.fromRowMajorWeights(null, "w", 2, 0, new float[0]));
        assertThrows(IllegalArgumentException.class,
                () -> DecoderOnlyWarpDenseProjection.fromRowMajorWeights(null, "w", 2, 3, new float[5]));
    }

    @Test
    void byteBufferOverloadMatchesFloatArrayOverloadOnWarp() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML");
        int outputSize = 2;
        int inputSize = 3;
        float[] weights = {
                1.0f, 2.0f, 3.0f,
                -1.0f, 0.5f, 4.0f
        };
        float[] input = {2.0f, -1.0f, 0.25f};

        // Read-only direct LE buffer mirrors a .wdmlpack mmap slice (the SmolLM2 H2c upload source).
        java.nio.ByteBuffer le = java.nio.ByteBuffer.allocateDirect(weights.length * Float.BYTES)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (float w : weights) {
            le.putFloat(w);
        }
        le.flip();
        java.nio.ByteBuffer readOnly = le.asReadOnlyBuffer().order(java.nio.ByteOrder.LITTLE_ENDIAN);

        try (WindowsBindings windowsBindings = new WindowsBindings()) {
            windowsBindings.init("warp");
            try (DecoderOnlyDenseProjection viaFloats = DecoderOnlyWarpDenseProjection.fromRowMajorWeights(
                    windowsBindings, "floats", outputSize, inputSize, weights);
                 DecoderOnlyDenseProjection viaBytes = DecoderOnlyWarpDenseProjection.fromRowMajorWeights(
                         windowsBindings, "bytes", outputSize, inputSize, readOnly)) {
                assertArrayEquals(viaFloats.project(input), viaBytes.project(input), 0.0f,
                        "ByteBuffer upload must be byte-identical to the float[] upload");
                assertEquals(0, readOnly.position());
                assertEquals(weights.length * Float.BYTES, readOnly.limit());
            }
        }
    }

    @Test
    void warpProjectionMatchesReferenceProjection() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML");
        float[] weights = {
                1.0f, 2.0f, 3.0f,
                -1.0f, 0.5f, 4.0f
        };
        float[] input = {2.0f, -1.0f, 0.25f};
        float[] reference = DecoderOnlyReferenceDenseProjection.fromRowMajorWeights(
                "reference", 2, 3, weights).project(input);

        try (WindowsBindings windowsBindings = new WindowsBindings()) {
            windowsBindings.init("warp");
            try (DecoderOnlyDenseProjection projection = DecoderOnlyWarpDenseProjection.fromRowMajorWeights(
                    windowsBindings, "warp", 2, 3, weights)) {
                float[] actual = projection.project(input);

                assertArrayEquals(reference, actual, 0.001f);
            }
        }
    }
}
