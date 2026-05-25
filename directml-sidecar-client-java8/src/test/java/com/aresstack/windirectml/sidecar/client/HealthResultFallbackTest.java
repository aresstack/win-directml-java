package com.aresstack.windirectml.sidecar.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Coverage for the dedicated fallback fields exposed by the
 * {@code health} method (see {@code docs/fallback-policy.md}).
 */
class HealthResultFallbackTest {

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void parsesAutoFallbackSelection() throws Exception {
        String json = "{\"ready\":true,"
                + "\"embeddingBackend\":\"cpu\",\"embeddingReady\":true,"
                + "\"embeddingFallback\":true,"
                + "\"embeddingFallbackReason\":\"DirectML.dll not found\","
                + "\"rerankerBackend\":\"cpu\",\"rerankerReady\":true,"
                + "\"rerankerFallback\":true,"
                + "\"rerankerFallbackReason\":\"no DML adapter\"}";
        JsonNode node = mapper.readTree(json);
        HealthResult h = HealthResult.from(node, json);

        assertEquals("cpu", h.getEmbeddingBackend());
        assertTrue(h.isEmbeddingFallback());
        assertEquals("DirectML.dll not found", h.getEmbeddingFallbackReason());

        assertEquals("cpu", h.getRerankerBackend());
        assertTrue(h.isRerankerFallback());
        assertEquals("no DML adapter", h.getRerankerFallbackReason());

        // lastError stays clean on soft fallback.
        assertNull(h.getLastError());
    }

    @Test
    void defaultsToFalseWhenFallbackFieldsAbsent() throws Exception {
        // Older sidecars may not emit the new fields; the client must
        // tolerate that and default to false / null.
        String json = "{\"ready\":true,\"embeddingBackend\":\"directml\",\"embeddingReady\":true}";
        JsonNode node = mapper.readTree(json);
        HealthResult h = HealthResult.from(node, json);

        assertEquals("directml", h.getEmbeddingBackend());
        assertFalse(h.isEmbeddingFallback());
        assertNull(h.getEmbeddingFallbackReason());
        assertFalse(h.isRerankerFallback());
        assertNull(h.getRerankerFallbackReason());
    }

    @Test
    void nullResultProducesNeutralHealth() {
        HealthResult h = HealthResult.from(null, "raw");
        assertFalse(h.isEmbeddingFallback());
        assertNull(h.getEmbeddingFallbackReason());
        assertFalse(h.isRerankerFallback());
        assertNull(h.getRerankerFallbackReason());
        assertEquals("raw", h.getRaw());
    }
}
