package com.aresstack.windirectml.sidecar.handlers;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
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
 * Handler for {@code embedBatch}.
 * <p>
 * Embeds multiple texts in a single JSON-RPC round-trip. When the
 * underlying {@link EmbeddingModel} is a bucket-batching backend
 * (e.g. {@code DirectMlBertEncoder}), this also coalesces the inputs
 * onto one DirectML dispatch per pad-bucket – the big perf hebel for
 * RAG-style ingestion and multi-chunk encoding.
 *
 * <pre>
 * params = {
 *   "texts":     ["...", "..."],    // required, non-empty
 *   "normalize": true,               // optional, default true (applies to all)
 *   "prefix":    "passage: "         // optional, applied to all (e.g. E5)
 * }
 * </pre>
 *
 * The reply preserves input order: {@code vectors[i]} corresponds to
 * {@code texts[i]}.
 *
 * <pre>
 * {
 *   "vectors":   [[..H..], [..H..], ...],
 *   "dimension": 384,
 *   "model":     "...",
 *   "normalized": true,
 *   "count":     N
 * }
 * </pre>
 */
public final class EmbedBatchHandler implements JsonRpcMethodHandler {

    private final Supplier<EmbeddingModel> modelSupplier;
    private final SidecarStatus status;

    public EmbedBatchHandler() {
        this(() -> null, null);
    }

    public EmbedBatchHandler(Supplier<EmbeddingModel> modelSupplier, SidecarStatus status) {
        this.modelSupplier = modelSupplier;
        this.status = status;
    }

    @Override
    public Object handle(JsonNode params) {
        if (status != null && status.isShuttingDown()) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.SHUTTING_DOWN, "Sidecar is shutting down");
        }
        EmbeddingModel model = modelSupplier != null ? modelSupplier.get() : null;
        if (model == null || !model.isReady()) {
            throw new JsonRpcMethodException(
                    JsonRpcErrorCode.NOT_IMPLEMENTED,
                    "embedBatch is not available: no encoder runtime registered.");
        }

        if (params == null || !params.isObject() || !params.hasNonNull("texts")) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.INVALID_PARAMS,
                    "params must be { \"texts\": [\"...\"] }");
        }
        JsonNode textsNode = params.get("texts");
        if (!textsNode.isArray() || textsNode.isEmpty()) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.INVALID_PARAMS,
                    "texts must be a non-empty array of strings");
        }
        boolean normalize = !params.hasNonNull("normalize") || params.get("normalize").asBoolean(true);
        String prefix = params.hasNonNull("prefix") ? params.get("prefix").asText() : null;

        List<EmbeddingRequest> requests = new ArrayList<>(textsNode.size());
        for (int i = 0; i < textsNode.size(); i++) {
            JsonNode t = textsNode.get(i);
            if (!t.isTextual()) {
                throw new JsonRpcMethodException(JsonRpcErrorCode.INVALID_PARAMS,
                        "texts[" + i + "] is not a string");
            }
            String text = t.asText();
            if (text.isBlank()) {
                throw new JsonRpcMethodException(JsonRpcErrorCode.INVALID_PARAMS,
                        "texts[" + i + "] must not be blank");
            }
            requests.add(new EmbeddingRequest(text, normalize, prefix));
        }

        if (status != null) status.setBusy(true);
        try {
            List<EmbeddingVector> vectors = model.embedBatch(requests);
            if (vectors.size() != requests.size()) {
                throw new JsonRpcMethodException(JsonRpcErrorCode.INTERNAL_ERROR,
                        "embedBatch returned " + vectors.size() + " vectors for "
                                + requests.size() + " requests");
            }
            List<float[]> arr = new ArrayList<>(vectors.size());
            String modelName = vectors.get(0).model();
            int dim = vectors.get(0).dimension();
            for (EmbeddingVector v : vectors) arr.add(v.values());

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("vectors", arr);
            result.put("dimension", dim);
            result.put("model", modelName);
            result.put("normalized", normalize);
            result.put("count", vectors.size());
            return result;
        } catch (EmbeddingException e) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.GENERATION_FAILED,
                    "Batch embedding failed: " + e.getMessage(), null, e);
        } catch (IllegalArgumentException e) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.INVALID_PARAMS, e.getMessage());
        } finally {
            if (status != null) status.setBusy(false);
        }
    }
}

