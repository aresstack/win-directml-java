package com.aresstack.windirectml.inference.decoderonly;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DecoderOnlyMathTest {

    @Test
    void normalizesWithRmsNorm() {
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] weight = {1.0f, 1.0f, 1.0f, 1.0f};

        DecoderOnlyMath.rmsNorm(values, weight, 1e-6f);

        float rms = (float) Math.sqrt(30.0 / 4.0 + 1e-6f);
        assertEquals(1.0f / rms, values[0], 1e-5f);
        assertEquals(4.0f / rms, values[3], 1e-5f);
    }

    @Test
    void calculatesStableSoftmax() {
        float[] values = {1.0f, 2.0f, 3.0f};

        DecoderOnlyMath.softmax(values, 3);

        assertEquals(1.0f, values[0] + values[1] + values[2], 1e-5f);
        assertTrue(values[2] > values[1]);
        assertTrue(values[1] > values[0]);
    }

    @Test
    void findsArgmax() {
        assertEquals(2, DecoderOnlyMath.argmax(new float[]{-1.0f, 0.5f, 0.75f, 0.1f}));
    }

    @Test
    void appliesRepetitionPenaltyOncePerTokenId() {
        float[] logits = {10.0f, -10.0f, 5.0f};
        int[] generated = {0, 0, 1};

        DecoderOnlyMath.applyRepetitionPenalty(logits, generated, 3, 2.0f);

        assertEquals(5.0f, logits[0], 1e-6f);
        assertEquals(-20.0f, logits[1], 1e-6f);
        assertEquals(5.0f, logits[2], 1e-6f);
    }
}
