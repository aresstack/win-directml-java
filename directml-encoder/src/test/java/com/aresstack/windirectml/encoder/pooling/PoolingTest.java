package com.aresstack.windirectml.encoder.pooling;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PoolingTest {

    @Test
    void meanPoolingIgnoresMaskedTokens() {
        float[][] tokens = {
                {1, 2, 3},
                {4, 5, 6},
                {99, 99, 99}, // masked padding
        };
        int[] mask = {1, 1, 0};
        float[] pooled = MeanPooling.pool(tokens, mask);
        assertArrayEquals(new float[]{2.5f, 3.5f, 4.5f}, pooled, 1e-6f);
    }

    @Test
    void meanPoolingWithAllMaskedReturnsZero() {
        float[][] tokens = {{1, 2}, {3, 4}};
        int[] mask = {0, 0};
        float[] pooled = MeanPooling.pool(tokens, mask);
        assertArrayEquals(new float[]{0f, 0f}, pooled, 1e-6f);
    }

    @Test
    void l2NormalizeProducesUnitVector() {
        float[] v = {3f, 4f};
        L2Normalize.inPlace(v, 1e-12f);
        double norm = Math.sqrt(v[0] * v[0] + v[1] * v[1]);
        assertEquals(1.0, norm, 1e-6);
        assertArrayEquals(new float[]{0.6f, 0.8f}, v, 1e-6f);
    }

    @Test
    void l2NormalizeHandlesZeroVector() {
        float[] v = {0f, 0f, 0f};
        L2Normalize.inPlace(v, 1e-6f);
        // No NaN/Inf, vector remains essentially zero
        for (float f : v) assertTrue(Float.isFinite(f));
    }
}

