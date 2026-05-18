package com.aresstack.windirectml.encoder.util;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CosineSimilarityTest {

    @Test
    void identicalVectorsHaveCosineOne() {
        float[] a = {1, 2, 3, 4};
        assertEquals(1.0, CosineSimilarity.compute(a, a), 1e-9);
    }

    @Test
    void orthogonalVectorsHaveCosineZero() {
        assertEquals(0.0, CosineSimilarity.compute(new float[]{1, 0}, new float[]{0, 1}), 1e-9);
    }

    @Test
    void oppositeVectorsHaveCosineMinusOne() {
        assertEquals(-1.0, CosineSimilarity.compute(new float[]{1, 2}, new float[]{-1, -2}), 1e-9);
    }

    @Test
    void zeroVectorReturnsZero() {
        assertEquals(0.0, CosineSimilarity.compute(new float[]{0, 0}, new float[]{1, 1}), 0.0);
    }
}

