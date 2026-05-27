package com.aresstack.windirectml.inference.qwen;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link Qwen2Runtime} math utilities.
 *
 * <p>Tests the fundamental building blocks (RMSNorm, softmax, argmax)
 * that the runtime uses. These do not require model weights.
 */
class Qwen2RuntimeMathTest {

    @Test
    void rmsNormNormalizesCorrectly() {
        // Simple test: x = [1, 2, 3, 4], weight = [1, 1, 1, 1], eps = 1e-6
        float[] x = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] weight = {1.0f, 1.0f, 1.0f, 1.0f};
        float eps = 1e-6f;

        Qwen2Runtime.rmsNorm(x, weight, eps);

        // RMS = sqrt((1+4+9+16)/4) = sqrt(30/4) = sqrt(7.5) ≈ 2.7386
        // Normalized: x[i] / RMS * weight[i]
        float rms = (float) Math.sqrt(30.0 / 4.0 + eps);
        float expected0 = 1.0f / rms;
        float expected1 = 2.0f / rms;
        float expected2 = 3.0f / rms;
        float expected3 = 4.0f / rms;

        assertEquals(expected0, x[0], 1e-5f);
        assertEquals(expected1, x[1], 1e-5f);
        assertEquals(expected2, x[2], 1e-5f);
        assertEquals(expected3, x[3], 1e-5f);
    }

    @Test
    void rmsNormWithWeightsScales() {
        float[] x = {1.0f, 1.0f, 1.0f, 1.0f};
        float[] weight = {2.0f, 0.5f, 1.0f, 3.0f};
        float eps = 1e-6f;

        Qwen2Runtime.rmsNorm(x, weight, eps);

        // RMS = sqrt((1+1+1+1)/4 + eps) = sqrt(1 + eps) ≈ 1.0
        // So normalized x[i] ≈ weight[i]
        assertEquals(2.0f, x[0], 1e-4f);
        assertEquals(0.5f, x[1], 1e-4f);
        assertEquals(1.0f, x[2], 1e-4f);
        assertEquals(3.0f, x[3], 1e-4f);
    }

    @Test
    void softmaxProducesProbabilityDistribution() {
        float[] x = {1.0f, 2.0f, 3.0f, 4.0f, 0.0f};

        Qwen2Runtime.softmax(x, 5);

        // Sum should be 1.0
        float sum = 0;
        for (int i = 0; i < 5; i++) sum += x[i];
        assertEquals(1.0f, sum, 1e-5f);

        // All values should be positive
        for (int i = 0; i < 5; i++) {
            assertTrue(x[i] > 0, "softmax output should be positive");
        }

        // Higher input → higher probability
        assertTrue(x[3] > x[2]);
        assertTrue(x[2] > x[1]);
        assertTrue(x[1] > x[0]);
        assertTrue(x[0] > x[4]);
    }

    @Test
    void softmaxWithSingleElement() {
        float[] x = {5.0f};
        Qwen2Runtime.softmax(x, 1);
        assertEquals(1.0f, x[0], 1e-6f);
    }

    @Test
    void softmaxPartialLen() {
        float[] x = {1.0f, 2.0f, 3.0f, 100.0f, 100.0f};
        Qwen2Runtime.softmax(x, 3);  // Only first 3 elements

        // Only first 3 should be a valid distribution
        float sum = x[0] + x[1] + x[2];
        assertEquals(1.0f, sum, 1e-5f);

        // Elements beyond len should remain unchanged
        assertEquals(100.0f, x[3], 1e-6f);
        assertEquals(100.0f, x[4], 1e-6f);
    }

    @Test
    void argmaxFindsMaximum() {
        float[] logits = {-1.0f, 3.5f, 2.0f, 0.5f, -2.0f};
        assertEquals(1, Qwen2Runtime.argmax(logits));
    }

    @Test
    void argmaxWithSingleElement() {
        float[] logits = {42.0f};
        assertEquals(0, Qwen2Runtime.argmax(logits));
    }

    @Test
    void argmaxWithNegativeValues() {
        float[] logits = {-5.0f, -3.0f, -10.0f, -1.0f, -7.0f};
        assertEquals(3, Qwen2Runtime.argmax(logits));
    }

    @Test
    void argmaxWithEqualMaxValues() {
        // When there are ties, should return the first occurrence
        float[] logits = {1.0f, 5.0f, 5.0f, 3.0f};
        assertEquals(1, Qwen2Runtime.argmax(logits));
    }

    @Test
    void kvHeadMappingUsesContiguousGroups() {
        // Qwen2.5-Coder 0.5B uses 14 query heads and 2 KV heads (7 query heads per KV head)
        int qHeadsPerKvHead = 7;
        int numKvHeads = 2;
        for (int h = 0; h < 14; h++) {
            int expectedKv = h < 7 ? 0 : 1;
            assertEquals(expectedKv, Qwen2Runtime.kvHeadForQueryHead(h, qHeadsPerKvHead, numKvHeads));
        }
    }

    @Test
    void kvHeadMappingRejectsOutOfRangeQueryHead() {
        assertThrows(IllegalArgumentException.class,
                () -> Qwen2Runtime.kvHeadForQueryHead(14, 7, 2));
    }
}
