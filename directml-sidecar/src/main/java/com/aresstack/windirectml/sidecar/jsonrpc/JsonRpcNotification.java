package com.aresstack.windirectml.sidecar.jsonrpc;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

/**
 * JSON-RPC 2.0 notification (no id, no expected response).
 * <p>
 * Used by the sidecar to push streaming chunks, progress, log events.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({"jsonrpc", "method", "params"})
public record JsonRpcNotification(
        @JsonProperty("jsonrpc") String jsonrpc,
        @JsonProperty("method") String method,
        @JsonProperty("params") Object params
) {
    public static JsonRpcNotification of(String method, Object params) {
        return new JsonRpcNotification(JsonRpcRequest.VERSION, method, params);
    }
}

