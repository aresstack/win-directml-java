package com.aresstack.windirectml.sidecar.client;

/**
 * Thrown when the sidecar returns a JSON-RPC error response.
 * Carries the JSON-RPC error code and the raw response line.
 */
public class JsonRpcError extends SidecarException {

    private static final long serialVersionUID = 1L;

    private final int code;
    private final String rawResponse;

    public JsonRpcError(int code, String message, String rawResponse) {
        super(message);
        this.code = code;
        this.rawResponse = rawResponse;
    }

    public int getCode() {
        return code;
    }

    public String getRawResponse() {
        return rawResponse;
    }
}

