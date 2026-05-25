package com.aresstack.windirectml.sidecar.client;

import com.fasterxml.jackson.databind.JsonNode;

/**
 * Typed view on the {@code health} response. Fields default to empty/false
 * when the sidecar does not yet expose them.
 */
public final class HealthResult {

    private final String status;
    private final boolean ready;
    private final boolean busy;
    private final boolean modelLoaded;
    private final boolean shuttingDown;
    private final String mode;
    private final String embeddingBackend;
    private final boolean embeddingReady;
    private final boolean embeddingFallback;
    private final String embeddingFallbackReason;
    private final String rerankerBackend;
    private final boolean rerankerReady;
    private final String rerankerModel;
    private final boolean rerankerFallback;
    private final String rerankerFallbackReason;
    private final boolean summarizerReady;
    private final String summarizerBackend;
    private final String summarizerModel;
    private final String lastError;
    private final String raw;

    private HealthResult(String status, boolean ready, boolean busy,
                         boolean modelLoaded, boolean shuttingDown,
                         String mode, String embeddingBackend,
                         boolean embeddingReady,
                         boolean embeddingFallback, String embeddingFallbackReason,
                         String rerankerBackend, boolean rerankerReady, String rerankerModel,
                         boolean rerankerFallback, String rerankerFallbackReason,
                         boolean summarizerReady, String summarizerBackend, String summarizerModel,
                         String lastError, String raw) {
        this.status = status;
        this.ready = ready;
        this.busy = busy;
        this.modelLoaded = modelLoaded;
        this.shuttingDown = shuttingDown;
        this.mode = mode;
        this.embeddingBackend = embeddingBackend;
        this.embeddingReady = embeddingReady;
        this.embeddingFallback = embeddingFallback;
        this.embeddingFallbackReason = embeddingFallbackReason;
        this.rerankerBackend = rerankerBackend;
        this.rerankerReady = rerankerReady;
        this.rerankerModel = rerankerModel;
        this.rerankerFallback = rerankerFallback;
        this.rerankerFallbackReason = rerankerFallbackReason;
        this.summarizerReady = summarizerReady;
        this.summarizerBackend = summarizerBackend;
        this.summarizerModel = summarizerModel;
        this.lastError = lastError;
        this.raw = raw;
    }

    public static HealthResult from(JsonNode result, String raw) {
        if (result == null) return new HealthResult(null, false, false, false, false,
                null, null, false, false, null, null, false, null,
                false, null, false, null, null, null, raw);
        return new HealthResult(
                asText(result, "status"),
                asBool(result, "ready"),
                asBool(result, "busy"),
                asBool(result, "modelLoaded"),
                asBool(result, "shuttingDown"),
                asText(result, "mode"),
                asText(result, "embeddingBackend"),
                asBool(result, "embeddingReady"),
                asBool(result, "embeddingFallback"),
                asText(result, "embeddingFallbackReason"),
                asText(result, "rerankerBackend"),
                asBool(result, "rerankerReady"),
                asText(result, "rerankerModel"),
                asBool(result, "rerankerFallback"),
                asText(result, "rerankerFallbackReason"),
                asBool(result, "summarizerReady"),
                asText(result, "summarizerBackend"),
                asText(result, "summarizerModel"),
                asText(result, "lastError"),
                raw);
    }

    private static String asText(JsonNode n, String k) {
        JsonNode v = n.get(k);
        return (v == null || v.isNull()) ? null : v.asText();
    }

    private static boolean asBool(JsonNode n, String k) {
        JsonNode v = n.get(k);
        return v != null && !v.isNull() && v.asBoolean(false);
    }

    public String getStatus() {
        return status;
    }

    public boolean isReady() {
        return ready;
    }

    public boolean isBusy() {
        return busy;
    }

    public boolean isModelLoaded() {
        return modelLoaded;
    }

    public boolean isShuttingDown() {
        return shuttingDown;
    }

    public String getMode() {
        return mode;
    }

    public String getEmbeddingBackend() {
        return embeddingBackend;
    }

    public boolean isEmbeddingReady() {
        return embeddingReady;
    }

    public String getLastError() {
        return lastError;
    }

    public String getRerankerBackend() { return rerankerBackend; }
    public boolean isRerankerReady() { return rerankerReady; }
    public String getRerankerModel() { return rerankerModel; }

    /**
     * {@code true} when the embedding backend was selected via {@code auto}
     * and the preferred backend (DirectML) was not available, so the sidecar
     * silently fell back to CPU. Use {@link #getEmbeddingFallbackReason()}
     * for the human-readable reason. Always {@code false} for forced modes
     * ({@code cpu}/{@code directml}) – those fail hard instead.
     */
    public boolean isEmbeddingFallback() { return embeddingFallback; }
    public String getEmbeddingFallbackReason() { return embeddingFallbackReason; }

    /** See {@link #isEmbeddingFallback()}; same semantics for the reranker. */
    public boolean isRerankerFallback() { return rerankerFallback; }
    public String getRerankerFallbackReason() { return rerankerFallbackReason; }

    public boolean isSummarizerReady() { return summarizerReady; }
    public String getSummarizerBackend() { return summarizerBackend; }
    public String getSummarizerModel() { return summarizerModel; }

    public String getRaw() {
        return raw;
    }
}
