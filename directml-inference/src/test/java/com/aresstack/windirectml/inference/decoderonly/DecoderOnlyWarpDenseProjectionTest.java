package com.aresstack.windirectml.inference.decoderonly;

import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
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
