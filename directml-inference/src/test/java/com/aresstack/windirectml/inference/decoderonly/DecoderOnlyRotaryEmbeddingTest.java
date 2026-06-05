package com.aresstack.windirectml.inference.decoderonly;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class DecoderOnlyRotaryEmbeddingTest {

    @Test
    void positionZeroLeavesVectorUnchanged() {
        DecoderOnlyRotaryEmbedding rope = new DecoderOnlyRotaryEmbedding(4, 10000.0, 2);
        float[] vector = {1.0f, 2.0f, 3.0f, 4.0f};

        rope.apply(vector, 0, 0);

        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, vector, 1e-6f);
    }

    @Test
    void usesOnTheFlyComputationPastPrecomputedRange() {
        DecoderOnlyRotaryEmbedding rope = new DecoderOnlyRotaryEmbedding(2, 10000.0, 1);
        float[] vector = {1.0f, 0.0f};

        rope.apply(vector, 0, 2);

        assertEquals((float) Math.cos(2.0), vector[0], 1e-6f);
        assertEquals((float) Math.sin(2.0), vector[1], 1e-6f);
    }
}
