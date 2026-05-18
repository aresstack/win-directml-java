package com.aresstack.windirectml.sidecar.jsonrpc;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

/**
 * Immutable JSON-RPC 2.0 error object.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({"code", "message", "data"})
public record JsonRpcError(
        @JsonProperty("code") int code,
        @JsonProperty("message") String message,
        @JsonProperty("data") Object data
) {
    public JsonRpcError(int code, String message) {
        this(code, message, null);
    }
}

