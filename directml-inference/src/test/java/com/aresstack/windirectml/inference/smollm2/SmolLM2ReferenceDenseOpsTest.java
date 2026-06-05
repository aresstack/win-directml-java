package com.aresstack.windirectml.inference.smollm2;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class SmolLM2ReferenceDenseOpsTest {

    @Test
    void dotMatchesScalarReference() {
        float[] left = {100.0f, 1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f, 7.0f, -8.0f};
        float[] right = {200.0f, 0.5f, 2.0f, -1.0f, 3.0f, 0.25f, -2.0f, 1.5f, 4.0f};

        float result = SmolLM2ReferenceDenseOps.dot(left, 1, right, 1, 8);

        assertEquals(-27.75f, result, 0.0001f);
    }

    @Test
    void multiplyRowsMatchesScalarReference() {
        float[] matrix = {
                1.0f, 2.0f, 3.0f,
                -1.0f, 0.5f, 4.0f
        };
        float[] input = {2.0f, -1.0f, 0.25f};
        float[] output = new float[2];

        SmolLM2ReferenceDenseOps.multiplyRows(matrix, 2, 3, input, output);

        assertArrayEquals(new float[]{0.75f, -1.5f}, output, 0.0001f);
    }

    @Test
    void gatedSiluMultiplyKeepsVectorShape() {
        float[] gate = {0.0f, 1.0f, -1.0f};
        float[] up = {2.0f, 3.0f, 4.0f};

        SmolLM2ReferenceDenseOps.gatedSiluMultiply(gate, up);

        assertEquals(0.0f, gate[0], 0.0001f);
        assertEquals(2.193f, gate[1], 0.01f);
        assertEquals(-1.075f, gate[2], 0.01f);
    }
}
