package com.aresstack.windirectml.sidecar.handlers;

import com.aresstack.windirectml.config.InputLimits;
import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.sidecar.SidecarStatus;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcErrorCode;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodException;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodHandler;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Handler für {@code embed}.
 * <p>
 * Erwartet eine optionale {@link EmbeddingModel}-Quelle. Solange keine
 * Encoder-Runtime registriert ist, antwortet der Handler mit
 * {@link JsonRpcErrorCode#NOT_IMPLEMENTED}; ist ein Modell vorhanden,
 * wird der Text eingebettet und der Vektor zurückgegeben.
 *
 * <pre>
 * params = { "text": "...", "normalize": true, "prefix": "query: " }
 * </pre>
 */
public final class EmbedHandler implements JsonRpcMethodHandler {

    private final Supplier<EmbeddingModel> modelSupplier;
    private final SidecarStatus status;

    public EmbedHandler() {
        this(() -> null, null);
    }

    public EmbedHandler(Supplier<EmbeddingModel> modelSupplier, SidecarStatus status) {
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
                    "embed is not available: no encoder runtime registered. "
                            + "Pending implementation – see milestone 5 (MiniLM).");
        }

        if (params == null || !params.isObject() || !params.hasNonNull("text")) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.INVALID_PARAMS,
                    "params must be { \"text\": \"...\" }");
        }
        String text = params.get("text").asText();
        if (text.isBlank()) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.INVALID_PARAMS,
                    "text must not be blank");
        }
        int maxLen = InputLimits.maxTextLength();
        if (text.length() > maxLen) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.LIMIT_EXCEEDED,
                    "text length " + text.length() + " exceeds maximum " + maxLen);
        }
        boolean normalize = !params.hasNonNull("normalize") || params.get("normalize").asBoolean(true);
        String prefix = params.hasNonNull("prefix") ? params.get("prefix").asText() : null;

        EmbeddingRequest req = new EmbeddingRequest(text, normalize, prefix);
        if (status != null) status.setBusy(true);
        try {
            EmbeddingVector vec = model.embed(req);
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("vector", vec.values());
            result.put("dimension", vec.dimension());
            result.put("model", vec.model());
            result.put("normalized", vec.normalized());
            return result;
        } catch (EmbeddingException e) {
            throw new JsonRpcMethodException(JsonRpcErrorCode.GENERATION_FAILED,
                    "Embedding failed: " + e.getMessage(), null, e);
        } finally {
            if (status != null) status.setBusy(false);
        }
    }
}
