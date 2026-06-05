package com.aresstack.windirectml.inference.smollm2;

/**
 * Signals that the SmolLM2 runtime API exists but execution is not implemented yet.
 */
public final class SmolLM2RuntimeUnsupportedException extends RuntimeException {
    public SmolLM2RuntimeUnsupportedException(String message) {
        super(message);
    }
}
