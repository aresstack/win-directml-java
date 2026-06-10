package com.aresstack.windirectml.inference.decoderonly;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DecoderOnlyReferenceDenseProjectionTest {

    @Test
    void projectsSingleVector() {
        DecoderOnlyDenseProjection projection = DecoderOnlyReferenceDenseProjection.fromRowMajorWeights(
                "test", 2, 3,
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        -1.0f, 0.5f, 4.0f
                });

        float[] output = projection.project(new float[]{2.0f, -1.0f, 0.25f});

        assertArrayEquals(new float[]{0.75f, -1.5f}, output, 0.0001f);
    }

    @Test
    void projectsSequenceWithDefaultBoundary() {
        DecoderOnlyDenseProjection projection = DecoderOnlyReferenceDenseProjection.fromRowMajorWeights(
                "test", 2, 2,
                new float[]{
                        1.0f, 0.0f,
                        0.0f, 2.0f
                });

        float[] output = projection.projectSequence(new float[]{
                1.0f, 2.0f,
                3.0f, 4.0f
        }, 2);

        assertArrayEquals(new float[]{1.0f, 4.0f, 3.0f, 8.0f}, output, 0.0001f);
    }

    @Test
    void rejectsInvalidWeightShape() {
        assertThrows(IllegalArgumentException.class, () -> DecoderOnlyReferenceDenseProjection.fromRowMajorWeights(
                "broken", 2, 3, new float[]{1.0f, 2.0f}));
    }
}
