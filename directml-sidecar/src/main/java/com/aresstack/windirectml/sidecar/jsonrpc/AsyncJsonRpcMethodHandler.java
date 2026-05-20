package com.aresstack.windirectml.sidecar.jsonrpc;

import com.fasterxml.jackson.databind.JsonNode;

/**
 * Extension of {@link JsonRpcMethodHandler} for handlers that need to
 * respond asynchronously (e.g. long-running Phi-3 inference).
 *
 * <p>When the {@link com.aresstack.windirectml.sidecar.SidecarCommandDispatcher}
 * detects that a registered handler implements this interface, it calls
 * {@link #handleAsync} instead of {@link #handle}. The handler is
 * responsible for writing exactly one {@link JsonRpcResponse} to
 * {@code writer} from any thread, and must not block the caller.
 *
 * <p>The {@link #handle} method inherited from {@link JsonRpcMethodHandler}
 * must throw {@link UnsupportedOperationException} – it will never be
 * called by the dispatcher for async handlers.
 */
public interface AsyncJsonRpcMethodHandler extends JsonRpcMethodHandler {

    /**
     * Start the asynchronous work. Must return immediately (non-blocking).
     * The implementation must eventually call {@code writer.writeResponse(…)}
     * exactly once with the given {@code id}.
     *
     * @param params the raw params node (may be null)
     * @param id     the request id node to echo in the response
     * @param writer thread-safe writer to use for the deferred response
     */
    void handleAsync(JsonNode params, JsonNode id, JsonRpcMessageWriter writer);

    /**
     * Not called for async handlers. Throws {@link UnsupportedOperationException}.
     */
    @Override
    default Object handle(JsonNode params) {
        throw new UnsupportedOperationException(
                "AsyncJsonRpcMethodHandler must be dispatched via handleAsync");
    }
}

