package com.aresstack.windirectml.inference.api;

import java.util.Objects;

/**
 * Typed runtime failure carrying a {@link GenerationErrorCode}. All failures surfaced by the public
 * generation API are instances of this exception so callers can branch on {@link #errorCode()}
 * without parsing messages.
 */
public final class GenerationException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    private final GenerationErrorCode errorCode;

    public GenerationException(GenerationErrorCode errorCode, String message) {
        this(errorCode, message, null);
    }

    public GenerationException(GenerationErrorCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = Objects.requireNonNull(errorCode, "errorCode");
    }

    public GenerationErrorCode errorCode() {
        return errorCode;
    }

    @Override
    public String getMessage() {
        return "[" + errorCode + "] " + super.getMessage();
    }
}
