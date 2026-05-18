package com.aresstack.windirectml.sidecar;

import com.aresstack.windirectml.inference.InferenceException;
import com.aresstack.windirectml.inference.Summarizer;
import com.aresstack.windirectml.inference.Summary;
import com.aresstack.windirectml.inference.SummaryRequest;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end test of the JSON-RPC sidecar lifecycle.
 * <p>
 * Drives {@link DirectMlPhi3Sidecar} in-process via piped streams, without
 * loading the real model. Verifies the contract from issue 21:
 * <ul>
 *   <li>Sidecar emits {@code sidecar.started} on startup.</li>
 *   <li>{@code health} works before model load.</li>
 *   <li>Invalid JSON does <em>not</em> crash the process.</li>
 *   <li>{@code shutdown} produces a response and ends the loop cleanly.</li>
 *   <li>stdout carries only JSON-RPC messages, one per line.</li>
 * </ul>
 */
class SidecarLifecycleTest {

    private final ObjectMapper mapper = new ObjectMapper();

    private static class StubSummarizer implements Summarizer {
        private final boolean ready;

        StubSummarizer(boolean ready) {
            this.ready = ready;
        }

        @Override
        public boolean isReady() {
            return ready;
        }

        @Override
        public Summary summarize(SummaryRequest request) throws InferenceException {
            return new Summary("stub: " + request.text(), "end_turn", 1, 1, 0L);
        }
    }

    private List<JsonNode> runWith(String input) throws Exception {
        ByteArrayInputStream in = new ByteArrayInputStream(input.getBytes(StandardCharsets.UTF_8));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        DirectMlPhi3Sidecar sidecar = new DirectMlPhi3Sidecar(
                in, out, Path.of("nonexistent"), "cpu", 128, false)
                .withSummarizer(new StubSummarizer(true));
        int exit = sidecar.run();
        assertEquals(0, exit);

        String text = out.toString(StandardCharsets.UTF_8);
        List<JsonNode> messages = new ArrayList<>();
        for (String line : text.split("\\R")) {
            if (line.isBlank()) continue;
            messages.add(mapper.readTree(line));
        }
        return messages;
    }

    @Test
    void healthBeforeModelLoadAndCleanShutdown() throws Exception {
        String input = """
                {"jsonrpc":"2.0","id":"h1","method":"health","params":{}}
                {"jsonrpc":"2.0","id":"x","method":"shutdown","params":{}}
                """;
        List<JsonNode> msgs = runWith(input);

        // 1. sidecar.started notification
        assertEquals("sidecar.started", msgs.get(0).path("method").asText());

        // 2. health response
        JsonNode health = msgs.get(1);
        assertEquals("h1", health.path("id").asText());
        JsonNode healthResult = health.path("result");
        assertTrue(healthResult.path("ready").asBoolean());
        assertEquals("ok", healthResult.path("status").asText());

        // 3. shutdown response
        JsonNode shutdown = msgs.get(2);
        assertEquals("x", shutdown.path("id").asText());
        assertTrue(shutdown.path("result").path("accepted").asBoolean());
    }

    @Test
    void invalidJsonDoesNotCrashProcess() throws Exception {
        String input = """
                {not-json-at-all
                {"jsonrpc":"2.0","id":"h","method":"health"}
                {"jsonrpc":"2.0","id":"x","method":"shutdown"}
                """;
        List<JsonNode> msgs = runWith(input);

        // first response after sidecar.started must be a parse error
        JsonNode parseErr = msgs.get(1);
        assertNotNull(parseErr.get("error"));
        assertEquals(-32700, parseErr.get("error").get("code").asInt());
        assertTrue(parseErr.get("id").isNull());

        // subsequent health works
        JsonNode health = msgs.get(2);
        assertEquals("h", health.path("id").asText());
        assertNull(health.get("error"));
    }

    @Test
    void unknownMethodReturnsMethodNotFound() throws Exception {
        String input = """
                {"jsonrpc":"2.0","id":"u","method":"unknownMethod"}
                {"jsonrpc":"2.0","id":"x","method":"shutdown"}
                """;
        List<JsonNode> msgs = runWith(input);

        JsonNode resp = msgs.get(1);
        assertEquals(-32601, resp.get("error").get("code").asInt());
    }

    @Test
    void summarizeUsesInjectedSummarizer() throws Exception {
        String input = """
                {"jsonrpc":"2.0","id":"s","method":"summarize","params":{"text":"hello"}}
                {"jsonrpc":"2.0","id":"x","method":"shutdown"}
                """;
        List<JsonNode> msgs = runWith(input);

        JsonNode resp = msgs.get(1);
        assertEquals("s", resp.path("id").asText());
        assertEquals("stub: hello", resp.path("result").path("text").asText());
    }

    @Test
    void embedReturnsNotImplementedWithoutEncoder() throws Exception {
        String input = """
                {"jsonrpc":"2.0","id":"e","method":"embed","params":{"text":"hi"}}
                {"jsonrpc":"2.0","id":"x","method":"shutdown"}
                """;
        List<JsonNode> msgs = runWith(input);

        JsonNode resp = msgs.get(1);
        assertEquals(-32005, resp.get("error").get("code").asInt());
    }

    @Test
    void stdoutIsOnlyJsonOnePerLine() throws Exception {
        ByteArrayInputStream in = new ByteArrayInputStream(
                "{\"jsonrpc\":\"2.0\",\"id\":\"x\",\"method\":\"shutdown\"}\n"
                        .getBytes(StandardCharsets.UTF_8));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        DirectMlPhi3Sidecar sidecar = new DirectMlPhi3Sidecar(
                in, out, Path.of("none"), "cpu", 128, false);
        sidecar.run();
        for (String line : out.toString(StandardCharsets.UTF_8).split("\\R")) {
            if (line.isBlank()) continue;
            assertTrue(line.startsWith("{") && line.endsWith("}"),
                    "stdout line is not a single JSON object: " + line);
            mapper.readTree(line); // must parse
        }
    }
}

