package com.aresstack.windirectml.sidecar.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JsonRpcResponseTest {

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void parsesResultResponse() throws Exception {
        String line = "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ready\":true,\"embeddingBackend\":\"directml\"}}";
        JsonRpcResponse r = JsonRpcResponse.parse(line, mapper);
        assertEquals(Long.valueOf(1L), r.getId());
        assertFalse(r.isError());
        assertFalse(r.isNotification());
        assertTrue(r.getResult().get("ready").asBoolean());
        assertEquals("directml", r.getResult().get("embeddingBackend").asText());

        HealthResult h = HealthResult.from(r.getResult(), r.getRaw());
        assertTrue(h.isReady());
        assertEquals("directml", h.getEmbeddingBackend());
    }

    @Test
    void parsesErrorResponse() throws Exception {
        String line = "{\"jsonrpc\":\"2.0\",\"id\":2,\"error\":{\"code\":-32005,\"message\":\"Not implemented\"}}";
        JsonRpcResponse r = JsonRpcResponse.parse(line, mapper);
        assertEquals(Long.valueOf(2L), r.getId());
        assertTrue(r.isError());
        assertEquals(-32005, r.getErrorCode());
        assertEquals("Not implemented", r.getErrorMessage());
    }

    @Test
    void parsesNotification() throws Exception {
        String line = "{\"jsonrpc\":\"2.0\",\"method\":\"sidecar.started\",\"params\":{\"name\":\"x\"}}";
        JsonRpcResponse r = JsonRpcResponse.parse(line, mapper);
        // The client only treats responses by id, so a missing id == notification.
        assertNull(r.getId());
        assertTrue(r.isNotification());
    }
}

