package com.aresstack.windirectml.inference.piper;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

final class PiperReductionOperationsTest {

    @Test
    void cumSumAccumulatesAlongRequestedTensorAxis() {
        PiperTensor input = PiperTensor.floats(new float[]{1, 2, 3, 4,
                5, 6, 7, 8}, 2, 2, 2);
        PiperTensor axis = PiperTensor.integers(new long[]{1});

        PiperTensor result = PiperReductionOperations.cumSum(input, axis, false, false);

        assertArrayEquals(new int[]{2, 2, 2}, result.shape());
        assertArrayEquals(new float[]{1, 2, 4, 6, 5, 6, 12, 14},
                result.floatData(), 0f);
    }

    @Test
    void cumSumSupportsExclusiveAndReverseIntegerSummation() {
        PiperTensor input = PiperTensor.integers(new long[]{1, 2, 3, 4}, 4);

        PiperTensor result = PiperReductionOperations.cumSum(
                input, PiperTensor.integers(new long[]{-1}, 1), true, true);

        assertArrayEquals(new long[]{9, 7, 4, 0}, result.integerData());
    }

    @Test
    void reducesSumsAndMaximumAcrossMiddleAxis() {
        PiperTensor input = PiperTensor.floats(new float[]{1, 2, 3, 4, 5, 6}, 1, 3, 2);

        PiperTensor sum = PiperReductionOperations.reduceSum(
                input, new long[]{1}, true, false);
        PiperTensor max = PiperReductionOperations.reduceMax(
                input, new long[]{1}, false, false);

        assertArrayEquals(new int[]{1, 1, 2}, sum.shape());
        assertArrayEquals(new float[]{9, 12}, sum.floatData(), 0f);
        assertArrayEquals(new int[]{1, 2}, max.shape());
        assertArrayEquals(new float[]{5, 6}, max.floatData(), 0f);
    }

    @Test
    void honorsNoopWithEmptyAxesAndRejectsInvalidIndices() {
        PiperTensor input = PiperTensor.integers(new long[]{2, 5}, 2);

        assertArrayEquals(input.integerData(),
                PiperReductionOperations.reduceSum(input, new long[0],
                        false, true).integerData());
        assertEquals(7, PiperReductionOperations.reduceSum(
                input, new long[0], false, false).integerData()[0]);
        assertThrows(IllegalArgumentException.class, () ->
                PiperReductionOperations.cumSum(input,
                        PiperTensor.integers(new long[]{2}), false, false));
        assertThrows(IllegalArgumentException.class, () ->
                PiperReductionOperations.reduceMax(input, new long[]{0, 0}, true, false));
    }
}
