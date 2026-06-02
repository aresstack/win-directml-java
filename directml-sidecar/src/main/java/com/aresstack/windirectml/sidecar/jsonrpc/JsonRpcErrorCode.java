package com.aresstack.windirectml.sidecar.jsonrpc;

/**
 * Standard JSON-RPC 2.0 error codes and sidecar-specific server error codes
 * (within the reserved -32000..-32099 range).
 */
public final class JsonRpcErrorCode {

    // ── JSON-RPC 2.0 standard codes ──────────────────────────────────────
    public static final int PARSE_ERROR = -32700;
    public static final int INVALID_REQUEST = -32600;
    public static final int METHOD_NOT_FOUND = -32601;
    public static final int INVALID_PARAMS = -32602;
    public static final int INTERNAL_ERROR = -32603;

    // ── Sidecar-specific server error codes ──────────────────────────────
    public static final int MODEL_NOT_READY = -32001;
    public static final int GENERATION_FAILED = -32002;
    public static final int SHUTTING_DOWN = -32003;
    public static final int CANCELLED = -32004;
    public static final int NOT_IMPLEMENTED = -32005;
    public static final int UNSUPPORTED_BACKEND = -32006;
    public static final int LIMIT_EXCEEDED = -32007;

    private JsonRpcErrorCode() {
    }
}

