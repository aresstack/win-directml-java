package com.aresstack.windirectml.sidecar.handlers;

import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodHandler;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Placeholder for the {@code cancel} method.
 * <p>
 * The current Phi-3 runtime is fully synchronous and processes one request
 * at a time, so there is nothing to cancel. The handler accepts the call
 * to keep the protocol surface stable; once streaming generation lands,
 * this will set a cancellation flag on the active inference job.
 */
public final class CancelHandler implements JsonRpcMethodHandler {

    @Override
    public Object handle(JsonNode params) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("accepted", false);
        result.put("reason", "no_active_request");
        return result;
    }
}

