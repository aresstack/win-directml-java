package com.aresstack.windirectml.sidecar.client;

import com.aresstack.windirectml.config.InputLimits;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * High-level Java-8 client that talks JSON-RPC 2.0 over stdin/stdout to a
 * Java-21 DirectML sidecar started via {@link SidecarProcess}.
 *
 * <p>The client maintains:
 * <ul>
 *   <li>a background reader thread that splits stdout into responses
 *       (keyed by {@code id}) and notifications (no {@code id});</li>
 *   <li>a registry of pending response futures, completed by id;</li>
 *   <li>last raw request / response strings for the JSON-RPC inspector.</li>
 * </ul>
 *
 * <p>Threading: blocking methods like {@link #health()} must be called from
 * a worker thread, never from a UI thread.
 */
public final class SidecarClient {

    private final SidecarClientConfig config;
    private final ObjectMapper mapper = new ObjectMapper();
    private final SidecarProcess process;
    private final AtomicLong idGen = new AtomicLong(1L);
    private final AtomicBoolean readerAlive = new AtomicBoolean(false);
    private final ConcurrentHashMap<Long, CompletableFuture<JsonRpcResponse>> pending =
            new ConcurrentHashMap<Long, CompletableFuture<JsonRpcResponse>>();
    private final LinkedBlockingDeque<JsonRpcResponse> notifications =
            new LinkedBlockingDeque<JsonRpcResponse>(256);

    private volatile String lastRawRequest;
    private volatile String lastRawResponse;
    private Thread readerThread;

    public SidecarClient(SidecarClientConfig config) {
        if (config == null) throw new IllegalArgumentException("config must not be null");
        this.config = config;
        this.process = new SidecarProcess(config);
    }

    // Convenience ctor for tests: inject a custom SidecarProcess subclass.
    SidecarClient(SidecarClientConfig config, SidecarProcess customProcess) {
        if (config == null) throw new IllegalArgumentException("config must not be null");
        this.config = config;
        this.process = customProcess;
    }

    // ── lifecycle ───────────────────────────────────────────────────────

    /**
     * Spawn the sidecar process and start the background reader thread.
     */
    public synchronized void start() throws SidecarException {
        process.start();
        readerAlive.set(true);
        readerThread = new Thread(new Runnable() {
            @Override
            public void run() {
                pumpStdout();
            }
        }, "sidecar-stdout-pump");
        readerThread.setDaemon(true);
        readerThread.start();
    }

    /**
     * Send a {@code shutdown} JSON-RPC request (best effort), then stop the
     * process. Always safe to call multiple times.
     */
    public synchronized void shutdown() {
        if (process.isRunning()) {
            try {
                sendRequestNoResponse("shutdown", null);
            } catch (SidecarException ignored) {
                // ignore – we are going to kill the process anyway
            }
        }
        readerAlive.set(false);
        process.stop(2000L);
        // Fail any pending futures with a clear error.
        for (Map.Entry<Long, CompletableFuture<JsonRpcResponse>> e : pending.entrySet()) {
            e.getValue().completeExceptionally(
                    new SidecarException("Sidecar shut down before response arrived"));
        }
        pending.clear();
    }

    public boolean isRunning() {
        return process.isRunning() && readerAlive.get();
    }

    public int exitValue() {
        return process.exitValue();
    }

    /**
     * Final exit code of the most recent stopped sidecar (snapshot taken
     * inside {@link SidecarProcess#stop(long)}). Returns {@code null}
     * when the process is still running or no exit code was observed.
     */
    public Integer lastExitCode() {
        return process.lastExitCode();
    }

    /**
     * {@code true} if the most recent stop had to fall back to
     * {@code destroyForcibly()}.
     */
    public boolean lastStopForced() {
        return process.lastStopForced();
    }

    public List<String> getCommandLine() {
        return process.getCommandLine();
    }

    // ── typed methods ───────────────────────────────────────────────────

    public HealthResult health() throws SidecarException {
        JsonRpcResponse resp = call("health", null);
        return HealthResult.from(resp.getResult(), resp.getRaw());
    }

    public EmbeddingResult embed(String text) throws SidecarException {
        return embed(text, true, null);
    }

    public EmbeddingResult embed(String text, boolean normalize, String prefix)
            throws SidecarException {
        if (text == null || text.length() == 0) {
            throw new SidecarException("embed: text must not be empty");
        }
        int maxLen = InputLimits.maxTextLength();
        if (text.length() > maxLen) {
            throw new SidecarException("embed: text length " + text.length()
                    + " exceeds maximum " + maxLen);
        }
        Map<String, Object> params = new LinkedHashMap<String, Object>();
        params.put("text", text);
        params.put("normalize", normalize);
        if (prefix != null) params.put("prefix", prefix);
        long t0 = System.currentTimeMillis();
        JsonRpcResponse resp = call("embed", params);
        long elapsed = System.currentTimeMillis() - t0;
        return EmbeddingResult.from(resp.getResult(), elapsed, resp.getRaw());
    }

    /**
     * Batch embedding. Sends all {@code texts} in a single JSON-RPC
     * round-trip; bucket-batching DirectML backends coalesce them onto
     * one GPU dispatch per pad-bucket. The result preserves input order.
     *
     * @param texts     non-null, non-empty list of non-blank texts.
     * @param normalize L2-normalise every vector (default {@code true}
     *                  in the JSON-RPC schema; pass explicitly here).
     * @param prefix    optional prefix prepended to every text
     *                  (e.g. {@code "passage: "} for E5).
     */
    public BatchEmbeddingResult embedBatch(List<String> texts, boolean normalize, String prefix)
            throws SidecarException {
        if (texts == null || texts.isEmpty()) {
            throw new SidecarException("embedBatch: texts must not be empty");
        }
        int maxBatch = InputLimits.maxEmbedBatchSize();
        if (texts.size() > maxBatch) {
            throw new SidecarException("embedBatch: batch size " + texts.size()
                    + " exceeds maximum " + maxBatch);
        }
        int maxLen = InputLimits.maxTextLength();
        for (int i = 0; i < texts.size(); i++) {
            String t = texts.get(i);
            if (t == null || t.trim().length() == 0) {
                throw new SidecarException("embedBatch: texts[" + i + "] must not be blank");
            }
            if (t.length() > maxLen) {
                throw new SidecarException("embedBatch: texts[" + i + "] length " + t.length()
                        + " exceeds maximum " + maxLen);
            }
        }
        Map<String, Object> params = new LinkedHashMap<String, Object>();
        params.put("texts", texts);
        params.put("normalize", normalize);
        if (prefix != null) params.put("prefix", prefix);
        long t0 = System.currentTimeMillis();
        JsonRpcResponse resp = call("embedBatch", params);
        long elapsed = System.currentTimeMillis() - t0;
        return BatchEmbeddingResult.from(resp.getResult(), elapsed, resp.getRaw());
    }

    /**
     * Convenience overload: {@code normalize=true}, no prefix.
     */
    public BatchEmbeddingResult embedBatch(List<String> texts) throws SidecarException {
        return embedBatch(texts, true, null);
    }

    public SummaryResult summarize(String text, int maxTokens) throws SidecarException {
        return summarize(text, maxTokens, null);
    }

    public SummaryResult summarize(String text, int maxTokens, String systemPrompt)
            throws SidecarException {
        if (text == null || text.length() == 0) {
            throw new SidecarException("summarize: text must not be empty");
        }
        Map<String, Object> params = new LinkedHashMap<String, Object>();
        params.put("text", text);
        if (maxTokens > 0) params.put("maxTokens", maxTokens);
        if (systemPrompt != null) params.put("systemPrompt", systemPrompt);
        long t0 = System.currentTimeMillis();
        // Use the dedicated summarize timeout (default 300 s) instead of the
        // general request timeout (default 30 s). Phi-3 inference is slow.
        JsonRpcResponse resp = callWithTimeout("summarize", params,
                config.getSummarizeTimeoutMillis());
        long elapsed = System.currentTimeMillis() - t0;
        return SummaryResult.from(resp.getResult(), elapsed, resp.getRaw());
    }

    /**
     * Cross-encoder reranking. Sends {@code (query, documents, topN)} and
     * returns the ranked list, already sorted by descending score.
     *
     * <p><b>Score semantics:</b> returned scores are raw classifier logits from
     * the loaded cross-encoder model. They are model-dependent, intended only
     * for relative ranking within the same query, and are not globally
     * calibrated probabilities. Do not compare scores across different models
     * or queries. See {@link RerankResult} for details.
     *
     * @param query     the search query.
     * @param documents candidate documents.
     * @param topN      maximum number of results to return; {@code <= 0}
     *                  returns all documents. Results are the top-N most
     *                  relevant after sorting.
     * @return ranked results sorted descending by score.
     * @throws SidecarException if local input validation fails, the sidecar
     *                          rejects the request, or the sidecar is not running.
     */
    public RerankResult rerank(String query, java.util.List<String> documents, int topN)
            throws SidecarException {
        if (query == null || query.length() == 0) {
            throw new SidecarException("rerank: query must not be empty");
        }
        int maxTextLen = InputLimits.maxTextLength();
        if (query.length() > maxTextLen) {
            throw new SidecarException("rerank: query length " + query.length()
                    + " exceeds maximum " + maxTextLen);
        }
        if (documents == null || documents.isEmpty()) {
            throw new SidecarException("rerank: documents must not be empty");
        }
        int maxDocs = InputLimits.maxRerankDocuments();
        if (documents.size() > maxDocs) {
            throw new SidecarException("rerank: documents count " + documents.size()
                    + " exceeds maximum " + maxDocs);
        }
        int maxDocLen = InputLimits.maxRerankDocumentLength();
        for (int i = 0; i < documents.size(); i++) {
            String d = documents.get(i);
            if (d != null && d.length() > maxDocLen) {
                throw new SidecarException("rerank: documents[" + i + "] length " + d.length()
                        + " exceeds maximum " + maxDocLen);
            }
        }
        Map<String, Object> params = new LinkedHashMap<String, Object>();
        params.put("query", query);
        params.put("documents", documents);
        if (topN > 0) params.put("topN", topN);
        long t0 = System.currentTimeMillis();
        JsonRpcResponse resp = call("rerank", params);
        long elapsed = System.currentTimeMillis() - t0;
        return RerankResult.from(resp.getResult(), elapsed, resp.getRaw());
    }

    // ── inspector accessors ─────────────────────────────────────────────

    public String getLastRawRequest() {
        return lastRawRequest;
    }

    public String getLastRawResponse() {
        return lastRawResponse;
    }

    public String getStderrSnapshot() {
        return process.getStderrSnapshot();
    }

    /**
     * Drain all notifications received so far (e.g. {@code sidecar.started}).
     */
    public java.util.List<JsonRpcResponse> drainNotifications() {
        java.util.List<JsonRpcResponse> out = new java.util.ArrayList<JsonRpcResponse>();
        notifications.drainTo(out);
        return out;
    }

    // ── core JSON-RPC request/response ──────────────────────────────────

    /**
     * Send a request and wait for the matching response, honouring
     * {@link SidecarClientConfig#getRequestTimeoutMillis()}.
     */
    public JsonRpcResponse call(String method, Object params) throws SidecarException {
        return callWithTimeout(method, params, config.getRequestTimeoutMillis());
    }

    /**
     * Send a request and wait for the matching response with an explicit timeout.
     */
    private JsonRpcResponse callWithTimeout(String method, Object params, long timeoutMillis)
            throws SidecarException {
        if (!isRunning()) {
            int code = process.exitValue();
            throw new SidecarException("Sidecar is not running (exit code "
                    + code + "). Recent stderr:\n" + process.getStderrSnapshot());
        }
        long id = idGen.getAndIncrement();
        JsonRpcRequest req = new JsonRpcRequest(id, method, params);
        String line = req.toJsonLine(mapper);
        CompletableFuture<JsonRpcResponse> future = new CompletableFuture<JsonRpcResponse>();
        pending.put(id, future);
        try {
            lastRawRequest = line;
            process.writeLine(line);
        } catch (IOException e) {
            pending.remove(id);
            throw new SidecarException("Failed to write JSON-RPC request: " + e.getMessage(), e);
        }
        JsonRpcResponse resp;
        try {
            resp = future.get(timeoutMillis, TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            pending.remove(id);
            throw new SidecarTimeoutException("Sidecar did not respond to '" + method
                    + "' within " + timeoutMillis + " ms");
        } catch (InterruptedException e) {
            pending.remove(id);
            Thread.currentThread().interrupt();
            throw new SidecarException("Interrupted while waiting for sidecar response", e);
        } catch (ExecutionException e) {
            pending.remove(id);
            Throwable cause = e.getCause();
            if (cause instanceof SidecarException) throw (SidecarException) cause;
            throw new SidecarException("Failure while reading sidecar response: "
                    + (cause == null ? e.getMessage() : cause.getMessage()), cause == null ? e : cause);
        }
        if (resp.isError()) {
            throw new JsonRpcError(resp.getErrorCode(), resp.getErrorMessage(), resp.getRaw());
        }
        return resp;
    }

    private void sendRequestNoResponse(String method, Object params) throws SidecarException {
        long id = idGen.getAndIncrement();
        JsonRpcRequest req = new JsonRpcRequest(id, method, params);
        String line = req.toJsonLine(mapper);
        try {
            lastRawRequest = line;
            process.writeLine(line);
        } catch (IOException e) {
            throw new SidecarException("Failed to write JSON-RPC request: " + e.getMessage(), e);
        }
    }

    private void pumpStdout() {
        while (readerAlive.get()) {
            String line;
            try {
                line = process.readLine();
            } catch (IOException e) {
                break;
            }
            if (line == null) break; // EOF
            line = line.trim();
            if (line.length() == 0) continue;
            JsonRpcResponse parsed;
            try {
                parsed = JsonRpcResponse.parse(line, mapper);
            } catch (SidecarException e) {
                // Malformed line; surface via lastRawResponse so the UI can see it.
                lastRawResponse = "<<malformed>> " + line;
                continue;
            }
            lastRawResponse = line;
            if (parsed.isNotification()) {
                // Bounded notification queue – drop oldest if full.
                if (!notifications.offerLast(parsed)) {
                    notifications.pollFirst();
                    notifications.offerLast(parsed);
                }
                continue;
            }
            Long id = parsed.getId();
            if (id == null) continue;
            CompletableFuture<JsonRpcResponse> f = pending.remove(id);
            if (f != null) f.complete(parsed);
        }
        // Reader stopped: fail outstanding requests so callers don't hang.
        readerAlive.set(false);
        for (Map.Entry<Long, CompletableFuture<JsonRpcResponse>> e : pending.entrySet()) {
            e.getValue().completeExceptionally(
                    new SidecarException("Sidecar stdout closed before response arrived"));
        }
        pending.clear();
    }

    // For test inspection only.
    ObjectMapper getMapperForTesting() {
        return mapper;
    }

    // Helper for unit tests that want to feed a synthetic response through
    // the same parsing path as the pump.
    static JsonRpcResponse parseLineForTesting(String line, ObjectMapper m) throws SidecarException {
        return JsonRpcResponse.parse(line, m);
    }

    // Build an ObjectNode quickly without going through ObjectMapper repeatedly.
    static ObjectNode emptyObject(ObjectMapper m) {
        return m.createObjectNode();
    }
}

