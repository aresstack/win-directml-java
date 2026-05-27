package com.aresstack.windirectml.config.generation;

/**
 * Thrown when a text-generation call fails.
 *
 * <p>Wraps backend-specific exceptions so callers of
 * {@link TextGenerationModel#generate(GenerationRequest)} do not need
 * to depend on Phi-3 or Qwen internals.
 *
 * <p>Java-8 compatible.
 */
public class GenerationException extends Exception {

    private static final long serialVersionUID = 1L;

    public GenerationException(String message) {
        super(message);
    }

    public GenerationException(String message, Throwable cause) {
        super(message, cause);
    }
}
