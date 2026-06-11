package com.aresstack.windirectml.inference.smollm2;

/**
 * Signals that a requested SmolLM2 runtime path cannot execute on this host — for example when the WARP device or
 * weight upload cannot be initialised, or when the runtime package is not executable. {@code AUTO} mode catches this
 * and falls back to the reference runtime; explicit {@code WARP} mode propagates it.
 */
public final class SmolLM2RuntimeUnsupportedException extends RuntimeException {
    public SmolLM2RuntimeUnsupportedException(String message) {
        super(message);
    }

    public SmolLM2RuntimeUnsupportedException(String message, Throwable cause) {
        super(message, cause);
    }
}
