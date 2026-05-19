package com.aresstack.windirectml.sidecar.handlers;

import com.aresstack.windirectml.encoder.reranker.RerankException;
import com.aresstack.windirectml.encoder.reranker.RerankRequest;
import com.aresstack.windirectml.encoder.reranker.RerankResult;
import com.aresstack.windirectml.encoder.reranker.Reranker;
import com.aresstack.windirectml.sidecar.SidecarStatus;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcErrorCode;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodException;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodHandler;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Handler for {@code rerank}.
 * <p>
 * Request shape:
 * <pre>
 * params = { "query": "...", "documents": ["...", "..."], "topN": 5 }
 * </pre>
 * Reply shape:
 * <pre>
 * { "model": "...", "results": [ {"index": 3, "score": 1.234}, ... ] }
 * </pre>
 * Results are sorted by descending score and trimmed to {@code topN}.
 * {@code topN <= 0} or {@code topN > documents.length} means "return all".
 * <p>
 * When no reranker model is registered the handler returns
 * {@link JsonRpcErrorCode#NOT_IMPLEMENTED}, matching the {@code embed}
 * handler's contract.
 */
public final class RerankHandler implements JsonRpcMethodHandler {

    private final Supplier<Reranker> rerankerSupplier;
    private final SidecarStatus status;

    public RerankHandler() {
        this(() -> null, null);
    }

    public RerankHandler(Supplier<Reranker> rerankerSupplier, SidecarStatus status) {
        this.rerankerSupplier = rerankerSupplier;
        this.status = status;
    }

    @Override
    public Object handle(JsonNode params) {
        if (status != null && status.isShuttingDown()) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.SHUTTING_DOWN, "Sidecar is shutting down");
        }
        Reranker reranker = rerankerSupplier != null ? rerankerSupplier.get() : null;
        if (reranker == null || !reranker.isReady()) {
            throw new JsonRpcMethodException(
                    JsonRpcErrorCode.NOT_IMPLEMENTED,
                    "rerank is not available: no reranker runtime registered. "
                            + "Provide -Drerank.modelDir or place a cross-encoder under model/.");
        }
        if (params == null || !params.isObject()
                || !params.hasNonNull("query") || !params.hasNonNull("documents")) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.INVALID_PARAMS,
                    "params must be { \"query\": \"...\", \"documents\": [\"...\"] }");
        }
        String query = params.get("query").asText();
        if (query.isBlank()) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.INVALID_PARAMS, "query must not be blank");
        }
        JsonNode docsNode = params.get("documents");
        if (!docsNode.isArray() || docsNode.isEmpty()) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.INVALID_PARAMS,
                    "documents must be a non-empty array of strings");
        }
        List<String> docs = new ArrayList<>(docsNode.size());
        for (int i = 0; i < docsNode.size(); i++) {
            JsonNode d = docsNode.get(i);
            if (!d.isTextual()) {
                throw new JsonRpcMethodException(JsonRpcErrorCode.INVALID_PARAMS,
                        "documents[" + i + "] is not a string");
            }
            docs.add(d.asText());
        }
        int topN = params.hasNonNull("topN") ? params.get("topN").asInt(0) : 0;

        if (status != null) status.setBusy(true);
        try {
            List<RerankResult> ranked = reranker.rerank(new RerankRequest(query, docs, topN));
            List<Map<String, Object>> items = new ArrayList<>(ranked.size());
            for (RerankResult r : ranked) {
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("index", r.originalIndex());
                item.put("score", r.score());
                items.add(item);
            }
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("model", reranker.modelName());
            result.put("results", items);
            return result;
        } catch (RerankException e) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.GENERATION_FAILED,
                    "Reranking failed: " + e.getMessage(), null, e);
        } catch (IllegalArgumentException e) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.INVALID_PARAMS, e.getMessage());
        } finally {
            if (status != null) status.setBusy(false);
        }
    }
}

