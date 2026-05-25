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
        // Wait briefly for any in-flight async handler (e.g. SummarizeHandler,
        // which offloads inference to a daemon worker thread) to finish
        // writing its response before we acknowledge shutdown. Async handlers
        // set `status.setBusy(true)` synchronously on the dispatch thread
        // before launching their worker, so by the time we get here `isBusy()`
        // reliably reflects pending async work. Bounded so a hung worker can't
        // block shutdown indefinitely.
        long deadlineNanos = System.nanoTime()
                + java.util.concurrent.TimeUnit.SECONDS.toNanos(5);
        while (status.isBusy() && System.nanoTime() < deadlineNanos) {
            try {
                Thread.sleep(5);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        status.setShuttingDown(true);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("accepted", true);
        result.put("status", "shutting_down");
        return result;
    }
}

