package com.aresstack.windirectml.runtime.facade;

/**
 * Thrown when an application attempts to load a model family that is not
 * yet supported by this runtime.
 * <p>
 * This exception is intentionally <b>explicit and non-silent</b>: callers
 * are informed clearly which family they requested and why it cannot be
 * loaded (planned/not-ready, unknown, etc.).
 */
public final class UnsupportedModelException extends RuntimeException {

    private final String family;

    public UnsupportedModelException(String family, String message) {
        super(message);
        this.family = family;
    }

    /**
     * The model family that was requested but is not supported.
     */
    public String family() {
        return family;
    }
}
