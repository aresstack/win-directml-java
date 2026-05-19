package com.aresstack.windirectml.sidecar.client;

/**
 * Base exception for all sidecar client failures. Java 8 compatible.
 */
public class SidecarException extends Exception {

    private static final long serialVersionUID = 1L;

    public SidecarException(String message) {
        super(message);
    }

    public SidecarException(String message, Throwable cause) {
        super(message, cause);
    }
}

