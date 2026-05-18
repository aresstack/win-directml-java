package com.aresstack.windirectml.sidecar.handlers;

import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcErrorCode;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodException;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodHandler;
import com.fasterxml.jackson.databind.JsonNode;

/**
 * Placeholder for the {@code embed} method.
 * <p>
 * Will be replaced once the encoder runtime (MiniLM/E5/JinaBERT) is online.
 * For now it returns a typed {@code NOT_IMPLEMENTED} error so the protocol
 * surface is stable and the Java-8 client can already wire the call site.
 */
public final class EmbedHandler implements JsonRpcMethodHandler {

    @Override
    public Object handle(JsonNode params) {
        throw new JsonRpcMethodException(
                JsonRpcErrorCode.NOT_IMPLEMENTED,
                "embed is not implemented yet. The encoder runtime "
                        + "(all-MiniLM-L6-v2 first) is tracked in milestone 5.");
    }
}

