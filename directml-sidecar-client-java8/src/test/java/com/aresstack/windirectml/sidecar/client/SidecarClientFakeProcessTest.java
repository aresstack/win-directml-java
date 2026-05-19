package com.aresstack.windirectml.sidecar.client;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * End-to-end test of {@link SidecarClient} against an in-memory fake
 * {@link SidecarProcess} that mimics the JSON-RPC framing without
 * actually spawning a JVM. Java-8 only, no GPU.
 */
class SidecarClientFakeProcessTest {

    /**
     * Drop-in replacement for {@link SidecarProcess} that pipes lines
     * through in-memory queues and lets the test script the responses.
     */
    static final class FakeSidecarProcess extends SidecarProcess {
        private final LinkedBlockingDeque<String> stdoutQueue = new LinkedBlockingDeque<String>();
        private final LinkedBlockingDeque<String> stdinQueue = new LinkedBlockingDeque<String>();
        private final AtomicBoolean alive = new AtomicBoolean(false);
        private final AtomicInteger exitCode = new AtomicInteger(-1);
        private final StringBuilder stderr = new StringBuilder();

        FakeSidecarProcess() {
            super(defaultCfg());
        }

        private static SidecarClientConfig defaultCfg() {
            SidecarClientConfig c = new SidecarClientConfig();
            c.setSidecarJarPath("fake");
            return c;
        }

        @Override
        public synchronized void start() {
            alive.set(true);
        }

        @Override
        public synchronized void writeLine(String line) throws IOException {
            if (!alive.get()) throw new IOException("fake process not running");
            stdinQueue.offerLast(line);
        }

        @Override
        public String readLine() throws IOException {
            try {
                String s = stdoutQueue.pollFirst(5, TimeUnit.SECONDS);
                if (s == null) {
                    return alive.get() ? null : null;
                }
                if (s.equals("<<EOF>>")) return null;
                return s;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("interrupted", e);
            }
        }

        @Override
        public synchronized void stop(long timeoutMillis) {
            alive.set(false);
            stdoutQueue.offerLast("<<EOF>>");
        }

        @Override
        public boolean isRunning() {
            return alive.get();
        }

        @Override
        public int exitValue() {
            return alive.get() ? -1 : exitCode.get();
        }

        @Override
        public String getStderrSnapshot() {
            return stderr.toString();
        }

        @Override
        public List<String> getCommandLine() {
            return java.util.Collections.singletonList("fake");
        }

        // Test helpers.
        String waitForRequest() throws InterruptedException {
            String s = stdinQueue.pollFirst(2, TimeUnit.SECONDS);
            assertNotNull(s, "expected sidecar to receive a request");
            return s;
        }

        void enqueueLine(String s) {
            stdoutQueue.offerLast(s);
        }

        void setExitCode(int code) {
            exitCode.set(code);
        }

        void appendStderr(String s) {
            stderr.append(s);
        }
    }

    private SidecarClient client;

    @AfterEach
    void cleanup() {
        if (client != null) client.shutdown();
    }

    @Test
    void healthRoundTripParsesResultAndCapturesRawTraffic() throws Exception {
        FakeSidecarProcess fake = new FakeSidecarProcess();
        SidecarClientConfig cfg = new SidecarClientConfig();
        cfg.setSidecarJarPath("fake");
        cfg.setRequestTimeoutMillis(2000L);
        client = new SidecarClient(cfg, fake);
        client.start();

        // Simulate the startup notification.
        fake.enqueueLine("{\"jsonrpc\":\"2.0\",\"method\":\"sidecar.started\",\"params\":{\"name\":\"fake\"}}");

        // Run health() on a worker thread so we can script the response.
        Thread t = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    HealthResult r = client.health();
                    assertEquals("directml", r.getEmbeddingBackend());
                    assertTrue(r.isReady());
                } catch (Exception e) {
                    fail("health() threw: " + e.getMessage());
                }
            }
        });
        t.start();

        String reqLine = fake.waitForRequest();
        assertTrue(reqLine.contains("\"method\":\"health\""));
        // Echo the id back as response.
        long id = client.getMapperForTesting().readTree(reqLine).get("id").asLong();
        fake.enqueueLine("{\"jsonrpc\":\"2.0\",\"id\":" + id
                + ",\"result\":{\"ready\":true,\"embeddingBackend\":\"directml\",\"embeddingReady\":true}}");
        t.join(3000L);

        assertNotNull(client.getLastRawRequest());
        assertNotNull(client.getLastRawResponse());
        // sidecar.started should be queued as a notification.
        List<JsonRpcResponse> notes = client.drainNotifications();
        assertEquals(1, notes.size());
        assertTrue(notes.get(0).isNotification());
    }

    @Test
    void timeoutSurfacesAsSidecarTimeoutException() throws Exception {
        FakeSidecarProcess fake = new FakeSidecarProcess();
        SidecarClientConfig cfg = new SidecarClientConfig();
        cfg.setSidecarJarPath("fake");
        cfg.setRequestTimeoutMillis(200L);
        client = new SidecarClient(cfg, fake);
        client.start();

        // Never enqueue a response. health() must throw a timeout.
        assertThrows(SidecarTimeoutException.class, new org.junit.jupiter.api.function.Executable() {
            @Override
            public void execute() throws Throwable {
                client.health();
            }
        });
    }

    @Test
    void jsonRpcErrorResponseIsRaisedAsTypedError() throws Exception {
        FakeSidecarProcess fake = new FakeSidecarProcess();
        SidecarClientConfig cfg = new SidecarClientConfig();
        cfg.setSidecarJarPath("fake");
        cfg.setRequestTimeoutMillis(2000L);
        client = new SidecarClient(cfg, fake);
        client.start();

        Thread t = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    client.embed("hello");
                    fail("expected JsonRpcError");
                } catch (JsonRpcError e) {
                    assertEquals(-32005, e.getCode());
                    assertTrue(e.getMessage().contains("not implemented")
                            || e.getMessage().toLowerCase().contains("not"));
                } catch (Exception e) {
                    fail("expected JsonRpcError, got " + e.getClass());
                }
            }
        });
        t.start();

        String reqLine = fake.waitForRequest();
        long id = client.getMapperForTesting().readTree(reqLine).get("id").asLong();
        fake.enqueueLine("{\"jsonrpc\":\"2.0\",\"id\":" + id
                + ",\"error\":{\"code\":-32005,\"message\":\"embed not implemented\"}}");
        t.join(3000L);
    }

    @Test
    void exitCodeIsReportedAfterShutdown() throws Exception {
        FakeSidecarProcess fake = new FakeSidecarProcess();
        SidecarClientConfig cfg = new SidecarClientConfig();
        cfg.setSidecarJarPath("fake");
        client = new SidecarClient(cfg, fake);
        client.start();
        assertTrue(client.isRunning());

        fake.setExitCode(3);
        client.shutdown();
        assertFalse(client.isRunning());
        assertEquals(3, client.exitValue());
    }

    @Test
    void rerankRoundTripParsesResult() throws Exception {
        FakeSidecarProcess fake = new FakeSidecarProcess();
        SidecarClientConfig cfg = new SidecarClientConfig();
        cfg.setSidecarJarPath("fake");
        cfg.setRequestTimeoutMillis(2000L);
        client = new SidecarClient(cfg, fake);
        client.start();

        final RerankResult[] captured = new RerankResult[1];
        final Throwable[] failure = new Throwable[1];
        Thread t = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    java.util.List<String> docs = java.util.Arrays.asList(
                            "first candidate document",
                            "second candidate document",
                            "third candidate document");
                    captured[0] = client.rerank("what is the query?", docs, 2);
                } catch (Throwable e) {
                    failure[0] = e;
                }
            }
        });
        t.start();

        String reqLine = fake.waitForRequest();
        // method must be "rerank"
        assertTrue(reqLine.contains("\"method\":\"rerank\""),
                "expected method=rerank, got: " + reqLine);
        // query/documents/topN must be serialised in params
        com.fasterxml.jackson.databind.JsonNode req = client.getMapperForTesting().readTree(reqLine);
        com.fasterxml.jackson.databind.JsonNode params = req.get("params");
        assertNotNull(params, "rerank request must have params");
        assertEquals("what is the query?", params.get("query").asText());
        com.fasterxml.jackson.databind.JsonNode docsNode = params.get("documents");
        assertNotNull(docsNode, "documents must be serialised");
        assertTrue(docsNode.isArray());
        assertEquals(3, docsNode.size());
        assertEquals("first candidate document", docsNode.get(0).asText());
        assertEquals("third candidate document", docsNode.get(2).asText());
        assertEquals(2, params.get("topN").asInt());

        // Reply with a scripted ranked result (already sorted by score desc on the sidecar).
        long id = req.get("id").asLong();
        fake.enqueueLine("{\"jsonrpc\":\"2.0\",\"id\":" + id
                + ",\"result\":{"
                + "\"model\":\"cross-encoder/ms-marco-MiniLM-L-6-v2\","
                + "\"results\":["
                + "{\"index\":2,\"score\":4.25},"
                + "{\"index\":0,\"score\":-0.5}"
                + "]}}");
        t.join(3000L);
        if (failure[0] != null) {
            fail("rerank() threw: " + failure[0].getMessage());
        }
        RerankResult r = captured[0];
        assertNotNull(r, "rerank() returned null");
        assertEquals("cross-encoder/ms-marco-MiniLM-L-6-v2", r.getModel());
        assertEquals(2, r.getItems().size());
        assertEquals(2, r.getItems().get(0).getIndex());
        assertEquals(4.25, r.getItems().get(0).getScore(), 1e-9);
        assertEquals(0, r.getItems().get(1).getIndex());
        assertEquals(-0.5, r.getItems().get(1).getScore(), 1e-9);
    }

    @Test
    void embedBatchRoundTripParsesVectorsInOrder() throws Exception {
        FakeSidecarProcess fake = new FakeSidecarProcess();
        SidecarClientConfig cfg = new SidecarClientConfig();
        cfg.setSidecarJarPath("fake");
        cfg.setRequestTimeoutMillis(2000L);
        client = new SidecarClient(cfg, fake);
        client.start();

        final BatchEmbeddingResult[] captured = new BatchEmbeddingResult[1];
        final Throwable[] failure = new Throwable[1];
        Thread t = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    java.util.List<String> texts = java.util.Arrays.asList(
                            "first chunk", "second chunk", "third chunk");
                    captured[0] = client.embedBatch(texts, true, "passage: ");
                } catch (Throwable e) {
                    failure[0] = e;
                }
            }
        });
        t.start();

        String reqLine = fake.waitForRequest();
        assertTrue(reqLine.contains("\"method\":\"embedBatch\""),
                "expected method=embedBatch, got: " + reqLine);

        com.fasterxml.jackson.databind.JsonNode req = client.getMapperForTesting().readTree(reqLine);
        com.fasterxml.jackson.databind.JsonNode params = req.get("params");
        assertNotNull(params, "embedBatch must have params");
        com.fasterxml.jackson.databind.JsonNode texts = params.get("texts");
        assertNotNull(texts, "texts must be serialised");
        assertTrue(texts.isArray());
        assertEquals(3, texts.size());
        assertEquals("first chunk", texts.get(0).asText());
        assertEquals("third chunk", texts.get(2).asText());
        assertTrue(params.get("normalize").asBoolean());
        assertEquals("passage: ", params.get("prefix").asText());

        long id = req.get("id").asLong();
        fake.enqueueLine("{\"jsonrpc\":\"2.0\",\"id\":" + id
                + ",\"result\":{"
                + "\"vectors\":[[1.0,2.0,3.0,4.0],[0.5,0.5,0.5,0.5],[-1.0,-2.0,-3.0,-4.0]],"
                + "\"dimension\":4,"
                + "\"model\":\"stub/embed\","
                + "\"normalized\":true,"
                + "\"count\":3"
                + "}}");
        t.join(3000L);
        if (failure[0] != null) {
            fail("embedBatch() threw: " + failure[0].getMessage());
        }

        BatchEmbeddingResult r = captured[0];
        assertNotNull(r, "embedBatch() returned null");
        assertEquals("stub/embed", r.getModel());
        assertEquals(4, r.getDimension());
        assertEquals(3, r.getCount());
        assertTrue(r.isNormalized());
        assertEquals(3, r.getVectors().size());
        assertEquals(1.0f, r.getVectors().get(0)[0], 1e-6f);
        assertEquals(0.5f, r.getVectors().get(1)[0], 1e-6f);
        assertEquals(-3.0f, r.getVectors().get(2)[2], 1e-6f);
    }

    @Test
    void embedBatchRejectsEmptyTextsLocally() throws Exception {
        FakeSidecarProcess fake = new FakeSidecarProcess();
        SidecarClientConfig cfg = new SidecarClientConfig();
        cfg.setSidecarJarPath("fake");
        client = new SidecarClient(cfg, fake);
        client.start();

        assertThrows(SidecarException.class, new org.junit.jupiter.api.function.Executable() {
            @Override
            public void execute() throws Throwable {
                client.embedBatch(java.util.Collections.<String>emptyList());
            }
        });
        assertThrows(SidecarException.class, new org.junit.jupiter.api.function.Executable() {
            @Override
            public void execute() throws Throwable {
                client.embedBatch(java.util.Arrays.asList("ok", ""));
            }
        });
        // Blank entries (whitespace-only) must also be rejected locally now.
        assertThrows(SidecarException.class, new org.junit.jupiter.api.function.Executable() {
            @Override
            public void execute() throws Throwable {
                client.embedBatch(java.util.Arrays.asList("ok", "   "));
            }
        });
    }
}

