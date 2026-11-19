package com.aresstack.windirectml.inference.piper;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

final class PiperIndexOperationsTest {

    @Test
    void scatterNdReplacesSelectedEntriesWithoutMutatingSource() {
        PiperTensor source = PiperTensor.floats(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        PiperTensor indices = PiperTensor.integers(new long[]{0, 1, 1, 2}, 2, 2);
        PiperTensor updates = PiperTensor.floats(new float[]{20, 60}, 2);

        PiperTensor result = PiperIndexOperations.scatterNd(source, indices, updates);

        assertArrayEquals(new int[]{2, 3}, result.shape());
        assertArrayEquals(new float[]{1, 20, 3, 4, 5, 60}, result.floatData(), 0f);
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, source.floatData(), 0f);
    }

    @Test
    void scatterAndGatherNdCopyWholeTrailingSlices() {
        PiperTensor source = PiperTensor.integers(new long[]{1, 2, 3, 4, 5, 6}, 2, 3);
        PiperTensor indices = PiperTensor.integers(new long[]{1, 0}, 2, 1);
        PiperTensor updates = PiperTensor.integers(new long[]{7, 8, 9, 10, 11, 12}, 2, 3);

        PiperTensor written = PiperIndexOperations.scatterNd(source, indices, updates);
        PiperTensor read = PiperIndexOperations.gatherNd(written,
                PiperTensor.integers(new long[]{-1, 0}, 2, 1));

        assertArrayEquals(new long[]{10, 11, 12, 7, 8, 9}, written.integerData());
        assertArrayEquals(new int[]{2, 3}, read.shape());
        assertArrayEquals(new long[]{7, 8, 9, 10, 11, 12}, read.integerData());
    }

    @Test
    void gatherElementsUsesIndicesAlongRequestedAxis() {
        PiperTensor input = PiperTensor.floats(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        PiperTensor indices = PiperTensor.integers(new long[]{0, -1, 1, 2}, 2, 2);

        PiperTensor output = PiperIndexOperations.gatherElements(input, indices, -1);

        assertArrayEquals(new int[]{2, 2}, output.shape());
        assertArrayEquals(new float[]{1, 3, 5, 6}, output.floatData(), 0f);
    }

    @Test
    void nonZeroReturnsCoordinateVectorsInRowMajorOrder() {
        PiperTensor flags = PiperTensor.bools(new boolean[]{false, true, false,
                true, false, false}, 2, 3);

        PiperTensor positions = PiperIndexOperations.nonZero(flags);

        assertArrayEquals(new int[]{2, 2}, positions.shape());
        assertArrayEquals(new long[]{0, 1, 1, 0}, positions.integerData());
    }

    @Test
    void rejectsOutOfBoundsAndInvalidScatterShapes() {
        PiperTensor data = PiperTensor.floats(new float[4], 2, 2);
        assertThrows(IllegalArgumentException.class, () ->
                PiperIndexOperations.scatterNd(data,
                        PiperTensor.integers(new long[]{2, 0}, 1, 2),
                        PiperTensor.floats(new float[]{4}, 1)));
        assertThrows(IllegalArgumentException.class, () ->
                PiperIndexOperations.gatherNd(data,
                        PiperTensor.integers(new long[]{-3, 0}, 1, 2)));
        assertThrows(IllegalArgumentException.class, () ->
                PiperIndexOperations.gatherElements(data,
                        PiperTensor.integers(new long[]{0, 4}, 1, 2), 1));
    }
}
