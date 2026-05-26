package com.aresstack.windirectml.runtime.api;

/**
 * Public rerank result returned by the Java 21 runtime API.
 *
 * @param originalIndex 0-based index into the caller-provided documents list.
 * @param score raw cross-encoder relevance score; higher means more relevant.
 */
public record RerankResult(int originalIndex, double score) {
    public RerankResult {
        if (originalIndex < 0) {
            throw new IllegalArgumentException("originalIndex < 0");
        }
    }

    static RerankResult fromInternal(com.aresstack.windirectml.encoder.reranker.RerankResult result) {
        return new RerankResult(result.originalIndex(), result.score());
    }
}
