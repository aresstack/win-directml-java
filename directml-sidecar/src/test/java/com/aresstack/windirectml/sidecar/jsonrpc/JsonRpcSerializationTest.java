package com.aresstack.windirectml.sidecar.jsonrpc;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JsonRpcSerializationTest {

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void deserializeRequest() throws Exception {
        String line = "{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"summarize\",\"params\":{\"text\":\"hi\"}}";
        JsonRpcRequest req = mapper.readValue(line, JsonRpcRequest.class);
        assertEquals("2.0", req.jsonrpc());
        assertEquals("summarize", req.method());
        assertEquals("1", req.id().asText());
        assertFalse(req.isNotification());
        assertTrue(req.isValid());
        JsonNode text = req.params().get("text");
        assertEquals("hi", text.asText());
    }

    @Test
    void deserializeNotification() throws Exception {
        String line = "{\"jsonrpc\":\"2.0\",\"method\":\"ping\"}";
        JsonRpcRequest req = mapper.readValue(line, JsonRpcRequest.class);
        assertTrue(req.isNotification());
        assertTrue(req.isValid());
        assertNull(req.params());
    }

    @Test
    void serializeSuccessResponseAsSingleLine() throws Exception {
        JsonRpcResponse resp = JsonRpcResponse.success(
                mapper.valueToTree("1"),
                Map.of("text", "summary"));
        String json = mapper.writeValueAsString(resp);
        assertFalse(json.contains("\n"));
        assertTrue(json.contains("\"jsonrpc\":\"2.0\""));
        assertTrue(json.contains("\"id\":\"1\""));
        assertTrue(json.contains("\"text\":\"summary\""));
        assertFalse(json.contains("\"error\""), "error must be omitted on success");
    }

    @Test
    void serializeErrorResponse() throws Exception {
        JsonRpcResponse resp = JsonRpcResponse.failure(
                mapper.valueToTree(42),
                new JsonRpcError(JsonRpcErrorCode.METHOD_NOT_FOUND, "Method not found: foo"));
        String json = mapper.writeValueAsString(resp);
        assertTrue(json.contains("\"id\":42"));
        assertTrue(json.contains("\"code\":-32601"));
        assertTrue(json.contains("\"message\":\"Method not found: foo\""));
        assertFalse(json.contains("\"result\""), "result must be omitted on error");
    }

    @Test
    void serializeNotification() throws Exception {
        JsonRpcNotification n = JsonRpcNotification.of("sidecar.modelLoaded",
                Map.of("loadTimeMs", 1234));
        String json = mapper.writeValueAsString(n);
        assertFalse(json.contains("\"id\""));
        assertTrue(json.contains("\"method\":\"sidecar.modelLoaded\""));
        assertTrue(json.contains("\"loadTimeMs\":1234"));
    }

    @Test
    void invalidRequestDetected() throws Exception {
        JsonRpcRequest noJsonRpc = mapper.readValue(
                "{\"id\":\"1\",\"method\":\"x\"}", JsonRpcRequest.class);
        assertFalse(noJsonRpc.isValid());

        JsonRpcRequest noMethod = mapper.readValue(
                "{\"jsonrpc\":\"2.0\",\"id\":\"1\"}", JsonRpcRequest.class);
        assertFalse(noMethod.isValid());
    }

    @Test
    void requestRoundTrip() throws Exception {
        String original = "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"health\",\"params\":{}}";
        JsonRpcRequest req = mapper.readValue(original, JsonRpcRequest.class);
        String back = mapper.writeValueAsString(req);
        assertNotNull(back);
        assertTrue(back.contains("\"method\":\"health\""));
        assertTrue(back.contains("\"id\":7"));
    }
}

