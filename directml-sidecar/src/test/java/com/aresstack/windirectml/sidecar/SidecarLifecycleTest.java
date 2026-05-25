package com.aresstack.windirectml.sidecar;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
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
import java.util.function.Consumer;

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
        return runWith(input, sidecar -> {
        });
    }

    private List<JsonNode> runWith(String input, Consumer<DirectMlPhi3Sidecar> configure) throws Exception {
        ByteArrayInputStream in = new ByteArrayInputStream(input.getBytes(StandardCharsets.UTF_8));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        DirectMlPhi3Sidecar sidecar = new DirectMlPhi3Sidecar(
                in, out, Path.of("nonexistent"), "cpu", 128, false)
                .withSummarizer(new StubSummarizer(true));
        configure.accept(sidecar);
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

    private static final class StubEmbeddingModel implements EmbeddingModel {
        private final String tag;

        StubEmbeddingModel(String tag) {
            this.tag = tag;
        }

        @Override
        public boolean isReady() {
            return true;
        }

        @Override
        public int dimension() {
            return 4;
        }

        @Override
        public EmbeddingVector embed(EmbeddingRequest r) throws EmbeddingException {
            return new EmbeddingVector(new float[]{1f, 0f, 0f, 0f}, 4, tag, r.normalize());
        }
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
    void summarizeReturnsNotImplementedWhenNoSummarizerConfigured() throws Exception {
        // Drive the sidecar without injecting any Summarizer. This is
        // the production fallback path when the Phi-3 model directory
        // is missing entirely – clients must still see a clean
        // -32005 NOT_IMPLEMENTED instead of a stack trace, matching
        // the embed / rerank "no model" behaviour.
        String input = """
                {"jsonrpc":"2.0","id":"s","method":"summarize","params":{"text":"hello"}}
                {"jsonrpc":"2.0","id":"x","method":"shutdown"}
                """;
        ByteArrayInputStream in = new ByteArrayInputStream(input.getBytes(StandardCharsets.UTF_8));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        DirectMlPhi3Sidecar sidecar = new DirectMlPhi3Sidecar(
                in, out, Path.of("nonexistent"), "cpu", 128, false);
        // intentionally no withSummarizer(...)
        assertEquals(0, sidecar.run());

        List<JsonNode> messages = new ArrayList<>();
        for (String line : out.toString(StandardCharsets.UTF_8).split("\\R")) {
            if (line.isBlank()) continue;
            messages.add(mapper.readTree(line));
        }

        JsonNode resp = messages.get(1);
        assertEquals("s", resp.path("id").asText());
        assertNotNull(resp.get("error"));
        assertEquals(-32005, resp.get("error").get("code").asInt());
        assertTrue(resp.get("error").get("message").asText().toLowerCase().contains("not implemented")
                || resp.get("error").get("message").asText().toLowerCase().contains("phi-3"));
    }

    @Test
    void summarizeReturnsNotImplementedWhenAutoLoadModelDirIsMissing() throws Exception {
        // Production path: autoLoadModel=true but the Phi-3 model
        // directory does not exist. The sidecar must NOT register a
        // real SummarizeHandler in this case – `summarize` should
        // answer with -32005 NOT_IMPLEMENTED, and the missing-file
        // diagnostic must be surfaced via `sidecar.modelLoadFailed`
        // and `health.lastError`.
        String input = """
                {"jsonrpc":"2.0","id":"s","method":"summarize","params":{"text":"hello"}}
                {"jsonrpc":"2.0","id":"h","method":"health"}
                {"jsonrpc":"2.0","id":"x","method":"shutdown"}
                """;
        ByteArrayInputStream in = new ByteArrayInputStream(input.getBytes(StandardCharsets.UTF_8));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        DirectMlPhi3Sidecar sidecar = new DirectMlPhi3Sidecar(
                in, out, Path.of("definitely-not-a-real-phi3-dir"), "cpu", 128, true);
        assertEquals(0, sidecar.run());

        List<JsonNode> messages = new ArrayList<>();
        for (String line : out.toString(StandardCharsets.UTF_8).split("\\R")) {
            if (line.isBlank()) continue;
            messages.add(mapper.readTree(line));
        }

        // Find the modelLoadFailed notification – it may be emitted
        // before sidecar.started.
        JsonNode loadFailed = null;
        for (JsonNode m : messages) {
            if ("sidecar.modelLoadFailed".equals(m.path("method").asText())) {
                loadFailed = m;
                break;
            }
        }
        assertNotNull(loadFailed, "expected sidecar.modelLoadFailed notification");
        String err = loadFailed.path("params").path("error").asText();
        assertTrue(err.toLowerCase().contains("phi-3"),
                "modelLoadFailed must include a Phi-3 diagnostic, got: " + err);
        assertTrue(err.contains("does not exist") || err.contains("missing"),
                "modelLoadFailed must explain why the dir is invalid, got: " + err);

        // summarize response must be -32005 NOT_IMPLEMENTED.
        JsonNode summarizeResp = null;
        JsonNode healthResp = null;
        for (JsonNode m : messages) {
            if ("s".equals(m.path("id").asText())) summarizeResp = m;
            if ("h".equals(m.path("id").asText())) healthResp = m;
        }
        assertNotNull(summarizeResp);
        assertNotNull(summarizeResp.get("error"));
        assertEquals(-32005, summarizeResp.get("error").get("code").asInt());

        // health.lastError must carry the same diagnostic.
        assertNotNull(healthResp);
        assertNotNull(healthResp.path("result").get("lastError"));
        assertTrue(healthResp.path("result").path("lastError").asText().toLowerCase().contains("phi-3"));
    }

    @Test
    void summarizeReturnsNotImplementedWhenAutoLoadModelDirIsIncomplete(@org.junit.jupiter.api.io.TempDir java.nio.file.Path tmp) throws Exception {
        // Production path with an *incomplete* Phi-3 model directory:
        // the missing file must be named in the diagnostic.
        java.nio.file.Files.writeString(tmp.resolve("config.json"), "{}");
        String input = """
                {"jsonrpc":"2.0","id":"s","method":"summarize","params":{"text":"hello"}}
                {"jsonrpc":"2.0","id":"x","method":"shutdown"}
                """;
        ByteArrayInputStream in = new ByteArrayInputStream(input.getBytes(StandardCharsets.UTF_8));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        DirectMlPhi3Sidecar sidecar = new DirectMlPhi3Sidecar(
                in, out, tmp, "cpu", 128, true);
        assertEquals(0, sidecar.run());

        List<JsonNode> messages = new ArrayList<>();
        for (String line : out.toString(StandardCharsets.UTF_8).split("\\R")) {
            if (line.isBlank()) continue;
            messages.add(mapper.readTree(line));
        }

        JsonNode loadFailed = null;
        JsonNode summarizeResp = null;
        for (JsonNode m : messages) {
            if ("sidecar.modelLoadFailed".equals(m.path("method").asText())) loadFailed = m;
            if ("s".equals(m.path("id").asText())) summarizeResp = m;
        }
        assertNotNull(loadFailed, "expected sidecar.modelLoadFailed notification");
        assertTrue(loadFailed.path("params").path("error").asText().contains("tokenizer.json"),
                "modelLoadFailed must name tokenizer.json, got: "
                        + loadFailed.path("params").path("error").asText());
        assertNotNull(summarizeResp);
        assertEquals(-32005, summarizeResp.get("error").get("code").asInt());
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
    void healthReportsEmbeddingBackendNoneWhenNoEncoderRegistered() throws Exception {
        String input = """
                {"jsonrpc":"2.0","id":"h","method":"health"}
                {"jsonrpc":"2.0","id":"x","method":"shutdown"}
                """;
        List<JsonNode> msgs = runWith(input);
        JsonNode result = msgs.get(1).path("result");
        assertEquals("none", result.path("embeddingBackend").asText());
        assertEquals(false, result.path("embeddingReady").asBoolean());
    }

    @Test
    void healthReportsCpuEmbeddingBackendWhenForced() throws Exception {
        String input = """
                {"jsonrpc":"2.0","id":"h","method":"health"}
                {"jsonrpc":"2.0","id":"e","method":"embed","params":{"text":"hi"}}
                {"jsonrpc":"2.0","id":"x","method":"shutdown"}
                """;
        List<JsonNode> msgs = runWith(input,
                s -> s.withEmbeddingBackend("cpu", new StubEmbeddingModel("cpu")));

        JsonNode health = msgs.get(1).path("result");
        assertEquals("cpu", health.path("embeddingBackend").asText());
        assertTrue(health.path("embeddingReady").asBoolean());

        JsonNode embed = msgs.get(2).path("result");
        assertEquals(4, embed.path("dimension").asInt());
        assertEquals("cpu", embed.path("model").asText());
    }

    @Test
    void healthReportsDirectMlEmbeddingBackendWhenForced() throws Exception {
        String input = """
                {"jsonrpc":"2.0","id":"h","method":"health"}
                {"jsonrpc":"2.0","id":"x","method":"shutdown"}
                """;
        List<JsonNode> msgs = runWith(input,
                s -> s.withEmbeddingBackend("directml", new StubEmbeddingModel("directml")));
        JsonNode result = msgs.get(1).path("result");
        assertEquals("directml", result.path("embeddingBackend").asText());
        assertTrue(result.path("embeddingReady").asBoolean());
    }

    @Test
    void autoFallbackResultIsObservableInHealth() throws Exception {
        // Simulate what main() does when EmbeddingBackendSelector returns
        // a fallback Selection: register CPU encoder, expose the dedicated
        // embeddingFallback signal (lastError is reserved for hard errors).
        String input = """
                {"jsonrpc":"2.0","id":"h","method":"health"}
                {"jsonrpc":"2.0","id":"x","method":"shutdown"}
                """;
        List<JsonNode> msgs = runWith(input, s -> {
            s.withEmbeddingBackend("cpu", new StubEmbeddingModel("cpu"));
            s.statusForTesting().setEmbeddingFallback(true);
            s.statusForTesting().setEmbeddingFallbackReason(
                    "embed.backend=auto: DirectML unavailable, falling back to CPU");
        });
        JsonNode result = msgs.get(1).path("result");
        assertEquals("cpu", result.path("embeddingBackend").asText());
        assertTrue(result.path("embeddingReady").asBoolean());
        assertTrue(result.path("embeddingFallback").asBoolean(),
                "embeddingFallback flag must be observable in health");
        assertNotNull(result.get("embeddingFallbackReason"));
        assertTrue(result.path("embeddingFallbackReason").asText().contains("auto"));
        // lastError stays clean when only a (soft) fallback happened.
        assertNull(result.get("lastError"));
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

