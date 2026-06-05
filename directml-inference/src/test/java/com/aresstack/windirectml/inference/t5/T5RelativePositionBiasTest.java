package com.aresstack.windirectml.inference.t5;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class T5RelativePositionBiasTest {
    @Test
    void usesMemoryPositionMinusQueryPositionForCausalBuckets() {
        T5RelativePositionBias bias = new T5RelativePositionBias(biasTable(32, 1), 32, 128, 1);

        float samePosition = bias.value(0, 2, 2, false);
        float previousToken = bias.value(0, 2, 1, false);
        float olderToken = bias.value(0, 2, 0, false);

        assertEquals(0.0f, samePosition);
        assertEquals(1.0f, previousToken);
        assertEquals(2.0f, olderToken);
    }

    @Test
    void usesMemoryPositionMinusQueryPositionForBidirectionalBuckets() {
        T5RelativePositionBias bias = new T5RelativePositionBias(biasTable(32, 1), 32, 128, 1);

        float previousToken = bias.value(0, 2, 1, true);
        float nextToken = bias.value(0, 2, 3, true);

        assertEquals(1.0f, previousToken);
        assertEquals(17.0f, nextToken);
    }

    private static T5TensorData biasTable(int buckets, int heads) {
        float[] values = new float[buckets * heads];
        for (int bucket = 0; bucket < buckets; bucket++) {
            values[bucket] = bucket;
        }
        return T5TensorData.reference("relative_attention_bias", new long[]{buckets, heads}, values);
    }
}
