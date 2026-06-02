package com.aresstack.windirectml.encoder.reranker;

/**
 * Checked failure during cross-encoder reranking
 * ({@link Reranker#rerank}). Wraps tokenisation, weight-loading and
 * GPU dispatch errors so callers see a single error category.
 */
public class RerankException extends Exception {
    private static final long serialVersionUID = 1L;

    public RerankException(String message) {
        super(message);
    }

    public RerankException(String message, Throwable cause) {
        super(message, cause);
    }
}

