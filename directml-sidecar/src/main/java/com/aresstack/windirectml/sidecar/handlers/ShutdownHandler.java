package com.aresstack.windirectml.sidecar.handlers;

import com.aresstack.windirectml.sidecar.SidecarStatus;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodHandler;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Handles the {@code shutdown} method.
 * <p>
 * Marks the sidecar as shutting down and returns an acknowledgement. The
 * main loop is expected to observe {@link SidecarStatus#isShuttingDown()}
 * and exit after writing the response.
 */
public final class ShutdownHandler implements JsonRpcMethodHandler {

    private final SidecarStatus status;

    public ShutdownHandler(SidecarStatus status) {
        this.status = status;
    }

    @Override
    public Object handle(JsonNode params) {
        status.setShuttingDown(true);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("accepted", true);
        result.put("status", "shutting_down");
        return result;
    }
}

