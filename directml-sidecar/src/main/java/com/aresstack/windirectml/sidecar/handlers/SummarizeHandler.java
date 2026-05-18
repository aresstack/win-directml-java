package com.aresstack.windirectml.sidecar.handlers;

import com.aresstack.windirectml.inference.InferenceException;
import com.aresstack.windirectml.inference.Summarizer;
import com.aresstack.windirectml.inference.Summary;
import com.aresstack.windirectml.inference.SummaryRequest;
import com.aresstack.windirectml.sidecar.SidecarStatus;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcErrorCode;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodException;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodHandler;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Handles the {@code summarize} method.
 * <pre>
 * params = { "text": "...", "maxTokens": 256, "systemPrompt": "..." }
 * </pre>
 */
public final class SummarizeHandler implements JsonRpcMethodHandler {

    private final Summarizer summarizer;
    private final SidecarStatus status;

    public SummarizeHandler(Summarizer summarizer, SidecarStatus status) {
        this.summarizer = summarizer;
        this.status = status;
    }

    @Override
    public Object handle(JsonNode params) {
        if (status.isShuttingDown()) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.SHUTTING_DOWN, "Sidecar is shutting down");
        }
        if (!summarizer.isReady()) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.MODEL_NOT_READY,
                    "Summarizer not ready (model still loading or load failed)");
        }
        if (params == null || !params.isObject()) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.INVALID_PARAMS,
                    "params must be an object with at least { \"text\": \"...\" }");
        }
        JsonNode textNode = params.get("text");
        if (textNode == null || !textNode.isTextual() || textNode.asText().isBlank()) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.INVALID_PARAMS,
                    "Missing or empty 'text' parameter");
        }

        int maxTokens = params.hasNonNull("maxTokens") ? params.get("maxTokens").asInt(0) : 0;
        String systemPrompt = params.hasNonNull("systemPrompt") ? params.get("systemPrompt").asText() : null;

        SummaryRequest request = new SummaryRequest(textNode.asText(), maxTokens, systemPrompt);

        status.setBusy(true);
        try {
            Summary summary = summarizer.summarize(request);
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("text", summary.text());
            result.put("finishReason", summary.finishReason());
            result.put("promptTokens", summary.promptTokens());
            result.put("outputTokens", summary.outputTokens());
            result.put("elapsedMs", summary.elapsedMs());
            return result;
        } catch (InferenceException e) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.GENERATION_FAILED,
                    "Summarization failed: " + e.getMessage(), null, e);
        } finally {
            status.setBusy(false);
        }
    }
}

