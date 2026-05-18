package com.aresstack.windirectml.sidecar;

import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcError;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcErrorCode;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodException;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodHandler;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcRequest;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcResponse;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.NullNode;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Dispatches incoming {@link JsonRpcRequest}s to registered method handlers.
 * <p>
 * Strictly isolates the JSON-RPC protocol layer from any inference-specific
 * details: handlers receive raw {@link JsonNode} parameters and return
 * arbitrary objects to be serialized by the shared {@code ObjectMapper}.
 * <p>
 * Unknown methods produce a standard {@code method not found} response;
 * any unchecked exception escaping a handler is translated to a generic
 * {@code internal error}. Handlers themselves should throw
 * {@link JsonRpcMethodException} for typed failures.
 */
public final class SidecarCommandDispatcher {

    private final Map<String, JsonRpcMethodHandler> handlers = new LinkedHashMap<>();

    public void register(String method, JsonRpcMethodHandler handler) {
        if (method == null || method.isBlank()) {
            throw new IllegalArgumentException("method must not be blank");
        }
        if (handler == null) {
            throw new IllegalArgumentException("handler must not be null");
        }
        handlers.put(method, handler);
    }

    public boolean hasHandler(String method) {
        return handlers.containsKey(method);
    }

    /**
     * Dispatch a parsed request.
     *
     * @return a fully-formed {@link JsonRpcResponse}. For notifications the
     * response is still returned but the caller should not write it
     * to stdout (id is null).
     */
    public JsonRpcResponse dispatch(JsonRpcRequest request) {
        JsonNode id = request.id() != null ? request.id() : NullNode.getInstance();

        if (!request.isValid()) {
            return JsonRpcResponse.failure(id,
                    new JsonRpcError(JsonRpcErrorCode.INVALID_REQUEST,
                            "Invalid JSON-RPC 2.0 request"));
        }

        JsonRpcMethodHandler handler = handlers.get(request.method());
        if (handler == null) {
            return JsonRpcResponse.failure(id,
                    new JsonRpcError(JsonRpcErrorCode.METHOD_NOT_FOUND,
                            "Method not found: " + request.method()));
        }

        try {
            Object result = handler.handle(request.params());
            return JsonRpcResponse.success(id, result);
        } catch (JsonRpcMethodException e) {
            return JsonRpcResponse.failure(id, e.toError());
        } catch (Exception e) {
            return JsonRpcResponse.failure(id,
                    new JsonRpcError(JsonRpcErrorCode.INTERNAL_ERROR,
                            "Internal error: " + e.getClass().getSimpleName()
                                    + ": " + e.getMessage()));
        }
    }
}

