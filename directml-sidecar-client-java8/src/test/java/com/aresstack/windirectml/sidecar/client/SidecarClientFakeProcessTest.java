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

    /** Drop-in replacement for {@link SidecarProcess} that pipes lines
     *  through in-memory queues and lets the test script the responses. */
    static final class FakeSidecarProcess extends SidecarProcess {
        private final LinkedBlockingDeque<String> stdoutQueue = new LinkedBlockingDeque<String>();
        private final LinkedBlockingDeque<String> stdinQueue  = new LinkedBlockingDeque<String>();
        private final AtomicBoolean alive = new AtomicBoolean(false);
        private final AtomicInteger exitCode = new AtomicInteger(-1);
        private final StringBuilder stderr = new StringBuilder();

        FakeSidecarProcess() { super(defaultCfg()); }

        private static SidecarClientConfig defaultCfg() {
            SidecarClientConfig c = new SidecarClientConfig();
            c.setSidecarJarPath("fake");
            return c;
        }

        @Override public synchronized void start() {
            alive.set(true);
        }
        @Override public synchronized void writeLine(String line) throws IOException {
            if (!alive.get()) throw new IOException("fake process not running");
            stdinQueue.offerLast(line);
        }
        @Override public String readLine() throws IOException {
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
        @Override public synchronized void stop(long timeoutMillis) {
            alive.set(false);
            stdoutQueue.offerLast("<<EOF>>");
        }
        @Override public boolean isRunning() { return alive.get(); }
        @Override public int exitValue() { return alive.get() ? -1 : exitCode.get(); }
        @Override public String getStderrSnapshot() { return stderr.toString(); }
        @Override public List<String> getCommandLine() {
            return java.util.Collections.singletonList("fake");
        }

        // Test helpers.
        String waitForRequest() throws InterruptedException {
            String s = stdinQueue.pollFirst(2, TimeUnit.SECONDS);
            assertNotNull(s, "expected sidecar to receive a request");
            return s;
        }
        void enqueueLine(String s) { stdoutQueue.offerLast(s); }
        void setExitCode(int code) { exitCode.set(code); }
        void appendStderr(String s) { stderr.append(s); }
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
            @Override public void run() {
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
            @Override public void execute() throws Throwable { client.health(); }
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
            @Override public void run() {
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
}

