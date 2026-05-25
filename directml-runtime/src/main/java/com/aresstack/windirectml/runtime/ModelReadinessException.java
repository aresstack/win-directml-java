package com.aresstack.windirectml.runtime;

/**
 * Thrown when an ML operation is invoked but the required model is not
 * configured or not ready.
 * <p>
 * This is an <em>unchecked</em> exception because model readiness is a
 * configuration/setup concern, not a transient runtime failure. Callers
 * should check {@link WinDirectMlRuntime#isEmbeddingReady()} or
 * {@link WinDirectMlRuntime#isRerankerReady()} upfront to avoid this.
 */
public class ModelReadinessException extends RuntimeException {

    public ModelReadinessException(String message) {
        super(message);
    }

    public ModelReadinessException(String message, Throwable cause) {
        super(message, cause);
    }
}
