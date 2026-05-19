package com.aresstack.windirectml.sidecar.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JsonRpcRequestTest {

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void serialisesHealthRequestWithEmptyParams() throws Exception {
        JsonRpcRequest req = new JsonRpcRequest(7L, "health", null);
        String line = req.toJsonLine(mapper);

        assertTrue(line.startsWith("{") && line.endsWith("}"),
                "line must be a single JSON object");
        assertEquals(-1, line.indexOf('\n'), "line must not contain newlines");

        JsonNode parsed = mapper.readTree(line);
        assertEquals("2.0", parsed.get("jsonrpc").asText());
        assertEquals(7, parsed.get("id").asLong());
        assertEquals("health", parsed.get("method").asText());
        assertTrue(parsed.has("params"));
        assertTrue(parsed.get("params").isObject());
        assertEquals(0, parsed.get("params").size());
    }

    @Test
    void serialisesEmbedRequestWithParams() throws Exception {
        Map<String, Object> params = new LinkedHashMap<String, Object>();
        params.put("text", "hello");
        params.put("normalize", true);
        JsonRpcRequest req = new JsonRpcRequest(42L, "embed", params);
        String line = req.toJsonLine(mapper);
        JsonNode parsed = mapper.readTree(line);
        assertEquals("embed", parsed.get("method").asText());
        assertEquals("hello", parsed.get("params").get("text").asText());
        assertEquals(true, parsed.get("params").get("normalize").asBoolean());
        assertEquals(42L, parsed.get("id").asLong());
    }

    @Test
    void summaryAndEmbeddingResultsParseFromMinimalJson() throws Exception {
        JsonNode embed = mapper.readTree(
                "{\"vector\":[1.0,0.0,0.0,0.0],\"dimension\":4,\"model\":\"x\",\"normalized\":true}");
        EmbeddingResult e = EmbeddingResult.from(embed, 5L, "raw");
        assertEquals(4, e.getDimension());
        assertEquals(4, e.getVector().length);
        assertEquals(1.0f, e.getVector()[0], 1e-9);
        assertEquals("x", e.getModel());
        assertTrue(e.isNormalized());

        JsonNode summary = mapper.readTree(
                "{\"text\":\"hi\",\"finishReason\":\"end_turn\",\"promptTokens\":3,\"outputTokens\":1,\"elapsedMs\":17}");
        SummaryResult s = SummaryResult.from(summary, 99L, "raw");
        assertEquals("hi", s.getText());
        assertEquals("end_turn", s.getFinishReason());
        assertEquals(3, s.getPromptTokens());
        assertEquals(1, s.getOutputTokens());
        // elapsedMs from the sidecar wins over the client-side measurement.
        assertEquals(17L, s.getElapsedMillis());
        assertNotNull(s.getRaw());
    }

    @Test
    void cosineComputesUnitVectors() {
        float[] a = {1f, 0f, 0f};
        float[] b = {1f, 0f, 0f};
        float[] c = {0f, 1f, 0f};
        assertEquals(1.0, EmbeddingResult.cosine(a, b), 1e-6);
        assertEquals(0.0, EmbeddingResult.cosine(a, c), 1e-6);
    }
}

