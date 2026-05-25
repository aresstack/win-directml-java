package com.aresstack.windirectml.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InputLimitsTest {

    @Test
    void defaultMaxTextLengthIs32768() {
        assertEquals(32_768, InputLimits.MAX_TEXT_LENGTH);
        assertEquals(32_768, InputLimits.maxTextLength());
    }

    @Test
    void defaultMaxEmbedBatchSizeIs256() {
        assertEquals(256, InputLimits.MAX_EMBED_BATCH_SIZE);
        assertEquals(256, InputLimits.maxEmbedBatchSize());
    }

    @Test
    void defaultMaxRerankDocumentsIs256() {
        assertEquals(256, InputLimits.MAX_RERANK_DOCUMENTS);
        assertEquals(256, InputLimits.maxRerankDocuments());
    }

    @Test
    void defaultMaxRerankDocumentLengthIs32768() {
        assertEquals(32_768, InputLimits.MAX_RERANK_DOCUMENT_LENGTH);
        assertEquals(32_768, InputLimits.maxRerankDocumentLength());
    }

    @Test
    void systemPropertyOverridesMaxTextLength() {
        String prop = InputLimits.PROP_MAX_TEXT_LENGTH;
        String old = System.getProperty(prop);
        try {
            System.setProperty(prop, "1024");
            assertEquals(1024, InputLimits.maxTextLength());
        } finally {
            if (old == null) System.clearProperty(prop);
            else System.setProperty(prop, old);
        }
    }

    @Test
    void invalidSystemPropertyFallsBackToDefault() {
        String prop = InputLimits.PROP_MAX_EMBED_BATCH_SIZE;
        String old = System.getProperty(prop);
        try {
            System.setProperty(prop, "not-a-number");
            assertEquals(InputLimits.MAX_EMBED_BATCH_SIZE, InputLimits.maxEmbedBatchSize());

            System.setProperty(prop, "-5");
            assertEquals(InputLimits.MAX_EMBED_BATCH_SIZE, InputLimits.maxEmbedBatchSize());

            System.setProperty(prop, "0");
            assertEquals(InputLimits.MAX_EMBED_BATCH_SIZE, InputLimits.maxEmbedBatchSize());
        } finally {
            if (old == null) System.clearProperty(prop);
            else System.setProperty(prop, old);
        }
    }

    @Test
    void allDefaultsArePositive() {
        assertTrue(InputLimits.maxTextLength() > 0);
        assertTrue(InputLimits.maxEmbedBatchSize() > 0);
        assertTrue(InputLimits.maxRerankDocuments() > 0);
        assertTrue(InputLimits.maxRerankDocumentLength() > 0);
    }
}
