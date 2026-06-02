package com.aresstack.windirectml.encoder.reranker;

import java.util.List;

/**
 * Cross-encoder reranker: scores {@code (query, document)} pairs with a
 * single forward pass through a BERT-style encoder followed by a
 * {@code [H,1]} classification head over the {@code [CLS]} token.
 * <p>
 * Implementations target different backends ({@link CpuReranker},
 * {@link DirectMlReranker}) but always expose this interface so the
 * sidecar can swap them transparently.
 */
public interface Reranker extends AutoCloseable {

    boolean isReady();

    /**
     * Human-readable model identifier (e.g. the HuggingFace repo name).
     */
    String modelName();

    /**
     * Score each {@code (query, doc)} pair and return the top results
     * sorted by descending {@link RerankResult#score()}.
     */
    List<RerankResult> rerank(RerankRequest request) throws RerankException;

    @Override
    void close();
}

