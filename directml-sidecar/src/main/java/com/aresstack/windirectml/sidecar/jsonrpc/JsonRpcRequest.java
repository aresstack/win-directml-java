package com.aresstack.windirectml.sidecar.jsonrpc;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import com.fasterxml.jackson.databind.JsonNode;

/**
 * Immutable JSON-RPC 2.0 request.
 * <p>
 * Notifications are recognized by {@link #id()} being {@code null}.
 * {@code params} may be {@code null}, an object node or an array node.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({"jsonrpc", "id", "method", "params"})
public record JsonRpcRequest(
        @JsonProperty("jsonrpc") String jsonrpc,
        @JsonProperty("id") JsonNode id,
        @JsonProperty("method") String method,
        @JsonProperty("params") JsonNode params
) {
    public static final String VERSION = "2.0";

    public boolean isNotification() {
        return id == null || id.isNull();
    }

    public boolean isValid() {
        return VERSION.equals(jsonrpc) && method != null && !method.isBlank();
    }
}

