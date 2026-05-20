package com.aresstack.windirectml.sidecar;

import com.aresstack.windirectml.sidecar.jsonrpc.AsyncJsonRpcMethodHandler;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcError;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcErrorCode;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMessageWriter;
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
 * Supports two handler kinds:
 * <ul>
 *   <li>{@link JsonRpcMethodHandler} – synchronous: {@code handle()} is called
 *       on the dispatch thread and the result is returned as a
 *       {@link JsonRpcResponse}.</li>
 *   <li>{@link AsyncJsonRpcMethodHandler} – asynchronous: {@code handleAsync()}
 *       is called, the handler is responsible for writing the response to the
 *       supplied {@link JsonRpcMessageWriter} from any thread. The dispatcher
 *       returns {@code null} to signal "response already handled".</li>
 * </ul>
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
     * Dispatch a parsed request (synchronous handlers only – legacy overload).
     *
     * @return a fully-formed {@link JsonRpcResponse}. For notifications the
     *         response is still returned but the caller should not write it
     *         to stdout (id is null).
     */
    public JsonRpcResponse dispatch(JsonRpcRequest request) {
        return dispatch(request, null);
    }

    /**
     * Dispatch a parsed request, optionally supporting async handlers.
     *
     * @param writer may be {@code null} if no async handlers are registered;
     *               must be non-null when async handlers are expected.
     * @return a fully-formed {@link JsonRpcResponse} for synchronous handlers,
     * or {@code null} when an async handler has taken ownership of
     * writing the response.
     */
    public JsonRpcResponse dispatch(JsonRpcRequest request, JsonRpcMessageWriter writer) {
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

        // Async handler: hand off to background thread, return null so the
        // run-loop skips writing a response (the handler will do it itself).
        if (handler instanceof AsyncJsonRpcMethodHandler && writer != null) {
            try {
                ((AsyncJsonRpcMethodHandler) handler).handleAsync(request.params(), id, writer);
            } catch (Exception e) {
                // handleAsync itself threw synchronously (e.g. validation).
                return JsonRpcResponse.failure(id,
                        new JsonRpcError(JsonRpcErrorCode.INTERNAL_ERROR,
                                "Async dispatch failed: " + e.getMessage()));
            }
            return null; // response will be written by the handler
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
