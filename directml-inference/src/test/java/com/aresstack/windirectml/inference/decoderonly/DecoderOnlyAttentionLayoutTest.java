package com.aresstack.windirectml.inference.decoderonly;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DecoderOnlyAttentionLayoutTest {

    @Test
    void mapsGroupedQueryHeadsToContiguousKvHeads() {
        DecoderOnlyAttentionLayout layout = new DecoderOnlyAttentionLayout(14, 2);

        for (int queryHead = 0; queryHead < 14; queryHead++) {
            int expectedKvHead = queryHead < 7 ? 0 : 1;
            assertEquals(expectedKvHead, layout.kvHeadForQueryHead(queryHead));
        }
    }

    @Test
    void rejectsInvalidGroupedQueryShape() {
        assertThrows(IllegalArgumentException.class, () -> new DecoderOnlyAttentionLayout(14, 3));
    }

    @Test
    void rejectsOutOfRangeQueryHead() {
        DecoderOnlyAttentionLayout layout = new DecoderOnlyAttentionLayout(14, 2);

        assertThrows(IllegalArgumentException.class, () -> layout.kvHeadForQueryHead(14));
    }
}
