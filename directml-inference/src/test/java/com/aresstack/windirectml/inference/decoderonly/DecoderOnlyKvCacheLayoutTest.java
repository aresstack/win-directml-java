package com.aresstack.windirectml.inference.decoderonly;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DecoderOnlyKvCacheLayoutTest {

    @Test
    void calculatesCanonicalFlatIndex() {
        DecoderOnlyKvCacheLayout layout = new DecoderOnlyKvCacheLayout(24, 2, 4096, 64);

        assertEquals(0, layout.flatIndex(0, 0, 0));
        assertEquals(63, layout.flatIndex(0, 0, 63));
        assertEquals(64, layout.flatIndex(0, 1, 0));
        assertEquals(4096 * 64, layout.flatIndex(1, 0, 0));
    }

    @Test
    void calculatesByteBudgets() {
        DecoderOnlyKvCacheLayout layout = new DecoderOnlyKvCacheLayout(24, 2, 4096, 64);

        assertEquals(2L * 4096L * 64L * Float.BYTES, layout.bytesPerLayer());
        assertEquals(2L * 24L * 2L * 4096L * 64L * Float.BYTES, layout.totalKeyValueBytes());
    }

    @Test
    void calculatesValidPrefixLength() {
        DecoderOnlyKvCacheLayout layout = new DecoderOnlyKvCacheLayout(1, 2, 16, 8);

        assertEquals(24, layout.validPrefixLength(3));
    }

    @Test
    void rejectsInvalidIndexes() {
        DecoderOnlyKvCacheLayout layout = new DecoderOnlyKvCacheLayout(1, 2, 16, 8);

        assertThrows(IllegalArgumentException.class, () -> layout.flatIndex(2, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> layout.flatIndex(0, 16, 0));
        assertThrows(IllegalArgumentException.class, () -> layout.flatIndex(0, 0, 8));
    }
}
