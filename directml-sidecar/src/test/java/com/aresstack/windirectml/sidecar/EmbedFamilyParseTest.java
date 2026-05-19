package com.aresstack.windirectml.sidecar;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Tests for {@link DirectMlPhi3Sidecar#embedFamily(String)} – the
 * stable parser for {@code -Dembed.model}.
 */
class EmbedFamilyParseTest {

    @Test
    void nullAndBlankDefaultToMinilm() {
        assertEquals("minilm", DirectMlPhi3Sidecar.embedFamily(null));
        assertEquals("minilm", DirectMlPhi3Sidecar.embedFamily(""));
        assertEquals("minilm", DirectMlPhi3Sidecar.embedFamily("   "));
    }

    @Test
    void recognisedAliasesMapToCanonicalToken() {
        assertEquals("minilm", DirectMlPhi3Sidecar.embedFamily("minilm"));
        assertEquals("minilm", DirectMlPhi3Sidecar.embedFamily("MiniLM"));
        assertEquals("minilm", DirectMlPhi3Sidecar.embedFamily("all-MiniLM-L6-v2"));
        assertEquals("e5", DirectMlPhi3Sidecar.embedFamily("e5"));
        assertEquals("e5", DirectMlPhi3Sidecar.embedFamily("E5"));
        assertEquals("e5", DirectMlPhi3Sidecar.embedFamily("e5-base-sts-en-de"));
    }

    @Test
    void unknownFamilyRejected() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> DirectMlPhi3Sidecar.embedFamily("jina"));
        org.junit.jupiter.api.Assertions.assertTrue(ex.getMessage().contains("e5"));
        org.junit.jupiter.api.Assertions.assertTrue(ex.getMessage().contains("minilm"));
    }
}

