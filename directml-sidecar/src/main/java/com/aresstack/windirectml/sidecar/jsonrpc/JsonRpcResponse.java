package com.aresstack.windirectml.sidecar.jsonrpc;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import com.fasterxml.jackson.databind.JsonNode;

/**
 * Immutable JSON-RPC 2.0 response.
 * <p>
 * Either {@link #result()} or {@link #error()} is set, never both.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({"jsonrpc", "id", "result", "error"})
public record JsonRpcResponse(
        @JsonProperty("jsonrpc") String jsonrpc,
        @JsonProperty("id") JsonNode id,
        @JsonProperty("result") Object result,
        @JsonProperty("error") JsonRpcError error
) {
    public static JsonRpcResponse success(JsonNode id, Object result) {
        return new JsonRpcResponse(JsonRpcRequest.VERSION, id, result, null);
    }

    public static JsonRpcResponse failure(JsonNode id, JsonRpcError error) {
        return new JsonRpcResponse(JsonRpcRequest.VERSION, id, null, error);
    }
}

