package com.aresstack.windirectml.inference.piper;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class PiperActivationOperationsTest {

    @Test
    void softplusIsStableForLargePositiveAndNegativeInputs() {
        PiperTensor input = PiperTensor.floats(new float[]{-100f, -1f, 0f, 1f, 100f}, 5);

        float[] output = PiperActivationOperations.softplus(input).floatData();

        for (float sample : output) {
            assertTrue(Float.isFinite(sample));
        }
        assertEquals(0f, output[0], 0.00001f);
        assertEquals(0.3132617f, output[1], 0.00001f);
        assertEquals(0.6931472f, output[2], 0.00001f);
        assertEquals(1.3132617f, output[3], 0.00001f);
        assertEquals(100f, output[4], 0.00001f);
    }

    @Test
    void leakyReluRespectsSlopeAndRetainsPositiveSamples() {
        PiperTensor input = PiperTensor.floats(new float[]{-4f, -2f, 0f, 2f}, 2, 2);

        PiperTensor output = PiperActivationOperations.leakyRelu(input, 0.1f);

        assertArrayEquals(new float[]{-0.4f, -0.2f, 0f, 2f}, output.floatData(), 0.000001f);
        assertArrayEquals(new int[]{2, 2}, output.shape());
    }

    @Test
    void clipHonorsScalarBoundsAndPreservesInput() {
        PiperTensor input = PiperTensor.floats(new float[]{-4f, -1f, 0f, 3f}, 4);
        PiperTensor min = PiperTensor.floats(new float[]{-2f});
        PiperTensor max = PiperTensor.floats(new float[]{2f});

        PiperTensor clipped = PiperActivationOperations.clip(input, min, max);

        assertArrayEquals(new float[]{-2f, -1f, 0f, 2f}, clipped.floatData(), 0f);
        assertArrayEquals(new float[]{-4f, -1f, 0f, 3f}, input.floatData(), 0f);
        assertArrayEquals(new float[]{-1f, 1f, 3f, 5f},
                PiperActivationOperations.ceil(PiperTensor.floats(
                        new float[]{-1.5f, 0.5f, 2.1f, 4.5f}, 4)).floatData(), 0f);
    }

    @Test
    void rejectsInvalidBoundsAndSlope() {
        PiperTensor input = PiperTensor.floats(new float[]{1f, 2f}, 2);
        assertThrows(IllegalArgumentException.class,
                () -> PiperActivationOperations.leakyRelu(input, Float.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> PiperActivationOperations.clip(input, PiperTensor.floats(new float[]{3f}),
                        PiperTensor.floats(new float[]{2f})));
    }
}
