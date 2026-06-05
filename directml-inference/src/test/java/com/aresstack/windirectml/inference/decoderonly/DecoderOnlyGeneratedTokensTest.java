package com.aresstack.windirectml.inference.decoderonly;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DecoderOnlyGeneratedTokensTest {

    @Test
    void storesTokensWithoutReallocation() {
        DecoderOnlyGeneratedTokens tokens = new DecoderOnlyGeneratedTokens(3);

        assertTrue(tokens.isEmpty());
        tokens.add(42);
        tokens.add(7);

        assertFalse(tokens.isEmpty());
        assertEquals(2, tokens.count());
        assertEquals(3, tokens.capacity());
        assertArrayEquals(new int[]{42, 7}, tokens.copyTokenIds());
    }

    @Test
    void rejectsOverflow() {
        DecoderOnlyGeneratedTokens tokens = new DecoderOnlyGeneratedTokens(1);
        tokens.add(1);

        assertThrows(IllegalStateException.class, () -> tokens.add(2));
    }
}
