package com.aresstack.windirectml.sidecar.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * Outgoing JSON-RPC 2.0 request envelope. Plain POJO, Java-8 compatible.
 */
public final class JsonRpcRequest {

    private final long id;
    private final String method;
    private final Object params;

    public JsonRpcRequest(long id, String method, Object params) {
        this.id = id;
        this.method = method;
        this.params = params;
    }

    public long getId() {
        return id;
    }

    public String getMethod() {
        return method;
    }

    public Object getParams() {
        return params;
    }

    /**
     * Serialise to a single-line JSON object using the provided mapper.
     */
    public String toJsonLine(ObjectMapper mapper) throws SidecarException {
        try {
            ObjectNode node = mapper.createObjectNode();
            node.put("jsonrpc", "2.0");
            node.put("id", id);
            node.put("method", method);
            if (params != null) {
                node.set("params", mapper.valueToTree(params));
            } else {
                node.putObject("params");
            }
            return mapper.writeValueAsString(node);
        } catch (RuntimeException e) {
            throw new SidecarException("Failed to serialise JSON-RPC request: " + e.getMessage(), e);
        } catch (Exception e) {
            throw new SidecarException("Failed to serialise JSON-RPC request: " + e.getMessage(), e);
        }
    }
}

