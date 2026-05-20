package com.aresstack.windirectml.sidecar.handlers;

import com.aresstack.windirectml.inference.InferenceException;
import com.aresstack.windirectml.inference.Summarizer;
import com.aresstack.windirectml.inference.Summary;
import com.aresstack.windirectml.inference.SummaryRequest;
import com.aresstack.windirectml.sidecar.SidecarStatus;
import com.aresstack.windirectml.sidecar.jsonrpc.AsyncJsonRpcMethodHandler;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcError;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcErrorCode;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMessageWriter;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodException;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcResponse;
import com.fasterxml.jackson.databind.JsonNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Handles the {@code summarize} method asynchronously.
 *
 * <p>Phi-3 inference can take 30–180 s depending on input length and
 * backend. Running it synchronously on the JSON-RPC dispatch thread would
 * block the entire sidecar for the full duration.  This handler implements
 * {@link AsyncJsonRpcMethodHandler}: validation is done on the dispatch
 * thread (fast fail), the actual ONNX inference is offloaded to a dedicated
 * daemon thread, and the response is written directly to the
 * {@link JsonRpcMessageWriter} (which is thread-safe) once the inference
 * finishes.
 *
 * <p>Only one summarize call is allowed at a time ({@code busy} flag).
 * Concurrent requests are rejected with {@code -32001 MODEL_NOT_READY}.
 */
public final class SummarizeHandler implements AsyncJsonRpcMethodHandler {

    private static final Logger log = LoggerFactory.getLogger(SummarizeHandler.class);

    private final Summarizer summarizer;
    private final SidecarStatus status;
    private final AtomicBoolean inferenceRunning = new AtomicBoolean(false);

    public SummarizeHandler(Summarizer summarizer, SidecarStatus status) {
        this.summarizer = summarizer;
        this.status = status;
    }

    @Override
    public void handleAsync(JsonNode params, JsonNode id, JsonRpcMessageWriter writer) {
        // ── fast validation on dispatch thread ───────────────────────────
        if (status.isShuttingDown()) {
            writer.writeResponse(JsonRpcResponse.failure(id,
                    new JsonRpcError(JsonRpcErrorCode.SHUTTING_DOWN,
                            "Sidecar is shutting down")));
            return;
        }
        if (!summarizer.isReady()) {
            writer.writeResponse(JsonRpcResponse.failure(id,
                    new JsonRpcError(JsonRpcErrorCode.MODEL_NOT_READY,
                            "Summarizer not ready (model still loading or load failed)")));
            return;
        }
        if (params == null || !params.isObject()) {
            writer.writeResponse(JsonRpcResponse.failure(id,
                    new JsonRpcError(JsonRpcErrorCode.INVALID_PARAMS,
                            "params must be an object with at least { \"text\": \"...\" }")));
            return;
        }
        JsonNode textNode = params.get("text");
        if (textNode == null || !textNode.isTextual() || textNode.asText().isBlank()) {
            writer.writeResponse(JsonRpcResponse.failure(id,
                    new JsonRpcError(JsonRpcErrorCode.INVALID_PARAMS,
                            "Missing or empty 'text' parameter")));
            return;
        }
        if (!inferenceRunning.compareAndSet(false, true)) {
            writer.writeResponse(JsonRpcResponse.failure(id,
                    new JsonRpcError(JsonRpcErrorCode.MODEL_NOT_READY,
                            "Another summarize request is already running")));
            return;
        }

        int maxTokens = params.hasNonNull("maxTokens") ? params.get("maxTokens").asInt(0) : 0;
        String systemPrompt = params.hasNonNull("systemPrompt")
                ? params.get("systemPrompt").asText() : null;
        SummaryRequest request = new SummaryRequest(textNode.asText(), maxTokens, systemPrompt);

        // ── offload inference to background thread ───────────────────────
        status.setBusy(true);
        Thread worker = new Thread(() -> {
            try {
                log.info("summarize: starting inference (maxTokens={}, textLen={})",
                        maxTokens > 0 ? maxTokens : "default",
                        textNode.asText().length());
                long t0 = System.currentTimeMillis();
                Summary summary = summarizer.summarize(request);
                long elapsed = System.currentTimeMillis() - t0;
                log.info("summarize: finished in {} ms (outputTokens={})",
                        elapsed, summary.outputTokens());
                Map<String, Object> result = new LinkedHashMap<>();
                result.put("text", summary.text());
                result.put("finishReason", summary.finishReason());
                result.put("promptTokens", summary.promptTokens());
                result.put("outputTokens", summary.outputTokens());
                result.put("elapsedMs", summary.elapsedMs());
                writer.writeResponse(JsonRpcResponse.success(id, result));
            } catch (InferenceException e) {
                log.error("summarize: inference failed: {}", e.getMessage(), e);
                writer.writeResponse(JsonRpcResponse.failure(id,
                        new JsonRpcError(JsonRpcErrorCode.GENERATION_FAILED,
                                "Summarization failed: " + e.getMessage())));
            } catch (Exception e) {
                log.error("summarize: unexpected error: {}", e.getMessage(), e);
                writer.writeResponse(JsonRpcResponse.failure(id,
                        new JsonRpcError(JsonRpcErrorCode.INTERNAL_ERROR,
                                "Unexpected error: " + e.getClass().getSimpleName()
                                        + ": " + e.getMessage())));
            } finally {
                inferenceRunning.set(false);
                status.setBusy(false);
            }
        }, "phi3-summarize-worker");
        worker.setDaemon(true);
        worker.start();
    }
}
