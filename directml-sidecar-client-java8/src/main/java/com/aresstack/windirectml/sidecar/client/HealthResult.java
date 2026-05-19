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
    private final String lastError;
    private final String raw;

    private HealthResult(String status, boolean ready, boolean busy,
                         boolean modelLoaded, boolean shuttingDown,
                         String mode, String embeddingBackend,
                         boolean embeddingReady, String lastError, String raw) {
        this.status = status;
        this.ready = ready;
        this.busy = busy;
        this.modelLoaded = modelLoaded;
        this.shuttingDown = shuttingDown;
        this.mode = mode;
        this.embeddingBackend = embeddingBackend;
        this.embeddingReady = embeddingReady;
        this.lastError = lastError;
        this.raw = raw;
    }

    public static HealthResult from(JsonNode result, String raw) {
        if (result == null) return new HealthResult(null, false, false, false, false,
                null, null, false, null, raw);
        return new HealthResult(
                asText(result, "status"),
                asBool(result, "ready"),
                asBool(result, "busy"),
                asBool(result, "modelLoaded"),
                asBool(result, "shuttingDown"),
                asText(result, "mode"),
                asText(result, "embeddingBackend"),
                asBool(result, "embeddingReady"),
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

    public String getStatus()             { return status; }
    public boolean isReady()              { return ready; }
    public boolean isBusy()               { return busy; }
    public boolean isModelLoaded()        { return modelLoaded; }
    public boolean isShuttingDown()       { return shuttingDown; }
    public String getMode()               { return mode; }
    public String getEmbeddingBackend()   { return embeddingBackend; }
    public boolean isEmbeddingReady()     { return embeddingReady; }
    public String getLastError()          { return lastError; }
    public String getRaw()                { return raw; }
}

