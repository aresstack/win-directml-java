package com.aresstack.windirectml.sidecar.client;

/**
 * Thrown when a JSON-RPC request does not get a matching response within
 * the configured timeout window.
 */
public class SidecarTimeoutException extends SidecarException {

    private static final long serialVersionUID = 1L;

    public SidecarTimeoutException(String message) {
        super(message);
    }
}

