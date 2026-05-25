package com.aresstack.windirectml.sidecar.handlers;

import com.aresstack.windirectml.sidecar.SidecarStatus;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodHandler;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Handles the {@code health} method.
 * <p>
 * Returns a snapshot of the sidecar status. Never touches the inference layer
 * – this is safe to call before the model has finished loading.
 */
public final class HealthHandler implements JsonRpcMethodHandler {

    private final SidecarStatus status;

    public HealthHandler(SidecarStatus status) {
        this.status = status;
    }

    @Override
    public Object handle(JsonNode params) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("status", status.isReady() ? "ok" : (status.isShuttingDown() ? "shutting_down" : "starting"));
        result.put("ready", status.isReady());
        result.put("busy", status.isBusy());
        result.put("modelLoaded", status.isModelLoaded());
        result.put("shuttingDown", status.isShuttingDown());
        if (status.getMode() != null) result.put("mode", status.getMode());
        result.put("embeddingBackend",
                status.getEmbeddingBackend() != null ? status.getEmbeddingBackend() : "none");
        result.put("embeddingReady", status.isEmbeddingReady());
        result.put("embeddingFallback", status.isEmbeddingFallback());
        if (status.getEmbeddingFallbackReason() != null) {
            result.put("embeddingFallbackReason", status.getEmbeddingFallbackReason());
        }
        result.put("rerankerBackend",
                status.getRerankerBackend() != null ? status.getRerankerBackend() : "none");
        result.put("rerankerReady", status.isRerankerReady());
        result.put("rerankerFallback", status.isRerankerFallback());
        if (status.getRerankerFallbackReason() != null) {
            result.put("rerankerFallbackReason", status.getRerankerFallbackReason());
        }
        if (status.getRerankerModel() != null) result.put("rerankerModel", status.getRerankerModel());
        // Phi-3 summarizer
        result.put("summarizerReady", status.isSummarizerReady());
        result.put("summarizerBackend",
                status.getSummarizerBackend() != null ? status.getSummarizerBackend() : "none");
        if (status.getSummarizerModel() != null) result.put("summarizerModel", status.getSummarizerModel());
        if (status.getLastError() != null) result.put("lastError", status.getLastError());
        return result;
    }
}
