package com.aresstack.windirectml.sidecar.jsonrpc;

/**
 * Runtime exception thrown by JSON-RPC method handlers to signal a
 * structured error back to the caller.
 */
public class JsonRpcMethodException extends RuntimeException {

    private final int code;
    private final Object data;

    public JsonRpcMethodException(int code, String message) {
        this(code, message, null, null);
    }

    public JsonRpcMethodException(int code, String message, Object data) {
        this(code, message, data, null);
    }

    public JsonRpcMethodException(int code, String message, Object data, Throwable cause) {
        super(message, cause);
        this.code = code;
        this.data = data;
    }

    public int code() { return code; }
    public Object data() { return data; }

    public JsonRpcError toError() {
        return new JsonRpcError(code, getMessage(), data);
    }
}

