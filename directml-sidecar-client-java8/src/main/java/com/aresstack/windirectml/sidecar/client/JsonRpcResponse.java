package com.aresstack.windirectml.sidecar.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Incoming JSON-RPC 2.0 response envelope (either {@code result} or
 * {@code error}). Java-8 compatible.
 */
public final class JsonRpcResponse {

    private final Long id;
    private final JsonNode result;
    private final JsonNode error;
    private final String raw;

    private JsonRpcResponse(Long id, JsonNode result, JsonNode error, String raw) {
        this.id = id;
        this.result = result;
        this.error = error;
        this.raw = raw;
    }

    public static JsonRpcResponse parse(String line, ObjectMapper mapper) throws SidecarException {
        try {
            JsonNode node = mapper.readTree(line);
            JsonNode idNode = node.get("id");
            Long parsedId = (idNode == null || idNode.isNull()) ? null : idNode.asLong();
            JsonNode result = node.get("result");
            JsonNode error  = node.get("error");
            return new JsonRpcResponse(parsedId, result, error, line);
        } catch (Exception e) {
            throw new SidecarException("Invalid JSON-RPC response line: " + line, e);
        }
    }

    public Long getId() {
        return id;
    }

    public boolean isNotification() {
        return id == null;
    }

    public boolean isError() {
        return error != null && !error.isNull();
    }

    public JsonNode getResult() {
        return result;
    }

    public JsonNode getError() {
        return error;
    }

    public String getRaw() {
        return raw;
    }

    public int getErrorCode() {
        if (!isError()) return 0;
        JsonNode codeNode = error.get("code");
        return codeNode == null ? 0 : codeNode.asInt();
    }

    public String getErrorMessage() {
        if (!isError()) return null;
        JsonNode msg = error.get("message");
        return msg == null ? null : msg.asText();
    }
}

