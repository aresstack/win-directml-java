package com.aresstack.windirectml.sidecar.jsonrpc;

import com.fasterxml.jackson.databind.JsonNode;

/**
 * A single JSON-RPC method handler.
 * <p>
 * Handlers receive the raw {@code params} JSON node (may be {@code null} or
 * a missing node) and return any value that can be serialized by the
 * shared {@link com.fasterxml.jackson.databind.ObjectMapper ObjectMapper}.
 * <p>
 * Failures must be reported via {@link JsonRpcMethodException} so the
 * dispatcher can produce a well-formed error response.
 */
@FunctionalInterface
public interface JsonRpcMethodHandler {

    Object handle(JsonNode params) throws JsonRpcMethodException;
}

