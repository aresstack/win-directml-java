package com.aresstack.windirectml.inference.piper;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

final class PiperTensorOperationsTest {

    @Test
    void gathersEmbeddingRowsAndPreservesBatchAndChannelOrder() {
        PiperTensor embeddings = PiperTensor.floats(
                new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
        PiperTensor ids = PiperTensor.integers(new long[]{2, 0}, 1, 2);

        PiperTensor gathered = PiperTensorOperations.gather(embeddings, ids, 0);
        PiperTensor transposed = PiperTensorOperations.transpose(gathered, List.of(0L, 2L, 1L));

        assertArrayEquals(new int[]{1, 2, 3}, gathered.shape());
        assertArrayEquals(new float[]{7, 8, 9, 1, 2, 3}, gathered.floatData(), 0f);
        assertArrayEquals(new int[]{1, 3, 2}, transposed.shape());
        assertArrayEquals(new float[]{7, 1, 8, 2, 9, 3}, transposed.floatData(), 0f);
    }

    @Test
    void createsSequenceMaskFromShapeGatherRangeAndUnsqueeze() {
        PiperTensor signal = PiperTensor.floats(new float[12], 1, 3, 4);
        PiperTensor actualLength = PiperTensor.integers(new long[]{3}, 1);

        PiperTensor length = PiperTensorOperations.gather(PiperTensorOperations.shape(signal),
                PiperTensor.integers(new long[]{2}), 0);
        PiperTensor positions = PiperTensorOperations.range(PiperTensor.integers(new long[]{0}),
                length, PiperTensor.integers(new long[]{1}));
        PiperTensor axis = PiperTensor.integers(new long[]{0}, 1);
        PiperTensor row = PiperTensorOperations.unsqueeze(positions, axis);
        PiperTensor column = PiperTensorOperations.unsqueeze(actualLength,
                PiperTensor.integers(new long[]{1}, 1));
        PiperTensor mask = PiperTensorOperations.binary("Less", row, column);
        PiperTensor expandedMask = PiperTensorOperations.unsqueeze(mask,
                PiperTensor.integers(new long[]{1}, 1));
        PiperTensor output = PiperTensorOperations.cast(expandedMask, PiperTensor.FLOAT32);

        assertArrayEquals(new long[]{0, 1, 2, 3}, positions.integerData());
        assertArrayEquals(new int[]{1, 1, 4}, output.shape());
        assertArrayEquals(new float[]{1f, 1f, 1f, 0f}, output.floatData(), 0f);
    }

    @Test
    void broadcastsAcrossChannelsAndRejectsInvalidDimensions() {
        PiperTensor values = PiperTensor.floats(new float[]{2, 4, 6, 8, 10, 12}, 1, 2, 3);
        PiperTensor mask = PiperTensor.floats(new float[]{1, 0, 1}, 1, 1, 3);

        PiperTensor output = PiperTensorOperations.binary("Mul", values, mask);

        assertArrayEquals(new float[]{2, 0, 6, 8, 0, 12}, output.floatData(), 0f);
        assertThrows(IllegalArgumentException.class, () -> PiperTensorOperations.binary(
                "Mul", values, PiperTensor.floats(new float[4], 1, 1, 4)));
        assertThrows(IllegalArgumentException.class, () -> PiperTensorOperations.gather(
                values, PiperTensor.integers(new long[]{5}), 2));
    }

    @Test
    void reshapesAndTransposesIntTensorsWithoutConvertingNumericType() {
        PiperTensor original = PiperTensor.integers(new long[]{1, 2, 3, 4}, 2, 2);
        PiperTensor transposed = PiperTensorOperations.transpose(original, List.of(1L, 0L));
        PiperTensor reshaped = PiperTensorOperations.reshape(transposed, 1, 4);

        assertArrayEquals(new long[]{1, 3, 2, 4}, reshaped.integerData());
        assertArrayEquals(new int[]{1, 4}, reshaped.shape());
        assertEquals(PiperTensor.INT64, reshaped.dataType());
    }

    @Test
    void concatsVectorsAndResolvesDynamicOnnxReshape() {
        PiperTensor vector = PiperTensorOperations.concat(List.of(
                PiperTensor.integers(new long[]{1}, 1),
                PiperTensor.integers(new long[]{2, 3}, 2)), 0);
        PiperTensor reshaped = PiperTensorOperations.reshapeOnnx(
                PiperTensor.floats(new float[]{1, 2, 3, 4, 5, 6}, 1, 2, 3),
                PiperTensor.integers(new long[]{0, -1}, 2), false);

        assertArrayEquals(new long[]{1, 2, 3}, vector.integerData());
        assertArrayEquals(new int[]{1, 6}, reshaped.shape());
        assertThrows(IllegalArgumentException.class, () -> PiperTensorOperations.reshapeOnnx(
                reshaped, PiperTensor.integers(new long[]{-1, -1}, 2), false));
    }

    @Test
    void multipliesBatchedAttentionMatrices() {
        PiperTensor query = PiperTensor.floats(
                new float[]{1, 2, 3, 4, 5, 6, 7, 8}, 1, 2, 2, 2);
        PiperTensor keys = PiperTensor.floats(new float[]{
                1, 0, 0, 1, 2, 0, 0, 2
        }, 1, 2, 2, 2);

        PiperTensor result = PiperTensorOperations.matmul(query, keys);

        assertArrayEquals(new int[]{1, 2, 2, 2}, result.shape());
        assertArrayEquals(new float[]{1, 2, 3, 4, 10, 12, 14, 16},
                result.floatData(), 0f);
    }

    @Test
    void slicesAndPadsRelativePositionEmbeddings() {
        PiperTensor values = PiperTensor.floats(
                new float[]{1, 2, 3, 4, 5}, 1, 5);
        PiperTensor sliced = PiperTensorOperations.slice(values,
                PiperTensor.integers(new long[]{1}),
                PiperTensor.integers(new long[]{4}),
                PiperTensor.integers(new long[]{1}),
                PiperTensor.integers(new long[]{1}));
        PiperTensor padded = PiperTensorOperations.pad(sliced,
                PiperTensor.integers(new long[]{0, 2, 0, 1}, 4));

        assertArrayEquals(new float[]{2, 3, 4}, sliced.floatData(), 0f);
        assertArrayEquals(new int[]{1, 6}, padded.shape());
        assertArrayEquals(new float[]{0, 0, 2, 3, 4, 0}, padded.floatData(), 0f);
    }

    @Test
    void appliesEqualWhereAndStableSoftmaxToAttentionScores() {
        PiperTensor scores = PiperTensor.floats(new float[]{1f, 2f, 1000f, 1000f}, 1, 2, 2);
        PiperTensor condition = PiperTensorOperations.equal(scores,
                PiperTensor.floats(new float[]{1000f}));
        PiperTensor filtered = PiperTensorOperations.where(condition,
                PiperTensor.floats(new float[]{-10000f}),
                scores);
        PiperTensor normalized = PiperTensorOperations.softmax(filtered, -1);

        assertArrayEquals(new boolean[]{false, false, true, true}, condition.booleanData());
        assertArrayEquals(new float[]{1f, 2f, -10000f, -10000f},
                filtered.floatData(), 0f);
        assertEquals(1f, normalized.floatData()[0] + normalized.floatData()[1], 0.00001f);
        assertEquals(0.5f, normalized.floatData()[2], 0.00001f);
        assertEquals(0.5f, normalized.floatData()[3], 0.00001f);
    }

    @Test
    void splitsChannelTensorsWithoutChangingTheirValues() {
        PiperTensor source = PiperTensor.floats(
                new float[]{1, 2, 3, 4, 5, 6, 7, 8}, 1, 4, 2);

        List<PiperTensor> parts = PiperTensorOperations.split(source, 1, new long[]{1, 3}, 2);

        assertArrayEquals(new int[]{1, 1, 2}, parts.get(0).shape());
        assertArrayEquals(new float[]{1, 2}, parts.get(0).floatData(), 0f);
        assertArrayEquals(new int[]{1, 3, 2}, parts.get(1).shape());
        assertArrayEquals(new float[]{3, 4, 5, 6, 7, 8}, parts.get(1).floatData(), 0f);
        assertThrows(IllegalArgumentException.class, () ->
                PiperTensorOperations.split(source, 1, new long[]{1, 2}, 2));
    }

    @Test
    void approximatesGaussianErrorFunctionWithinReferenceTolerance() {
        PiperTensor values = PiperTensor.floats(new float[]{-2, -1, 0, 1, 2}, 5);

        float[] evaluated = PiperTensorOperations.unary("Erf", values).floatData();

        assertArrayEquals(new float[]{-0.9953223f, -0.8427008f, 0f, 0.8427008f, 0.9953223f},
                evaluated, 0.000002f);
    }

    @Test
    void rangeSupportsFloatStepsUsedByAlignmentGraph() {
        PiperTensor ascending = PiperTensorOperations.range(
                PiperTensor.floats(new float[]{0f}),
                PiperTensor.floats(new float[]{1.5f}),
                PiperTensor.floats(new float[]{0.5f}));
        PiperTensor descending = PiperTensorOperations.range(
                PiperTensor.floats(new float[]{2f}),
                PiperTensor.floats(new float[]{-1f}),
                PiperTensor.floats(new float[]{-1f}));

        assertArrayEquals(new float[]{0f, 0.5f, 1f}, ascending.floatData(), 0f);
        assertArrayEquals(new float[]{2f, 1f, 0f}, descending.floatData(), 0f);
        assertThrows(IllegalArgumentException.class, () -> PiperTensorOperations.range(
                PiperTensor.floats(new float[]{0f}),
                PiperTensor.floats(new float[]{1f}),
                PiperTensor.floats(new float[]{0f})));
    }

    @Test
    void rejectsReshapeAndCastOutsideValidOnnxContract() {
        PiperTensor values = PiperTensor.floats(new float[]{1, 2, 3}, 3);

        assertThrows(IllegalArgumentException.class,
                () -> PiperTensorOperations.reshape(values, 2, 2));
        assertThrows(IllegalArgumentException.class,
                () -> PiperTensorOperations.cast(values, 999));
        assertThrows(IllegalArgumentException.class,
                () -> PiperTensorOperations.unsqueeze(values,
                        PiperTensor.integers(new long[]{0, 0}, 2)));
    }
}
