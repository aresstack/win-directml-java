package com.aresstack.windirectml.encoder;

/**
 * Failure raised by an {@link EmbeddingModel}.
 */
public class EmbeddingException extends Exception {

    public EmbeddingException(String message) {
        super(message);
    }

    public EmbeddingException(String message, Throwable cause) {
        super(message, cause);
    }
}

