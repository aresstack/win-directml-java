package com.aresstack.windirectml.runtime.facade;

import com.aresstack.windirectml.encoder.reranker.RerankException;
import com.aresstack.windirectml.encoder.reranker.RerankRequest;
import com.aresstack.windirectml.encoder.reranker.RerankResult;
import com.aresstack.windirectml.encoder.reranker.Reranker;

import java.util.List;
import java.util.Objects;

/**
 * High-level reranker model handle returned by
 * {@link LocalMlRuntime#loadRerankerModel(RerankerModelConfig)}.
 * <p>
 * Wraps the internal {@link Reranker} interface and provides a simple
 * API for scoring (query, document) pairs.
 */
public final class LocalRerankerModel implements AutoCloseable {

    private final Reranker delegate;

    LocalRerankerModel(Reranker delegate) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
    }

    /**
     * Score each document against the query and return results sorted
     * by descending relevance score.
     *
     * @param query     the search query.
     * @param documents candidate documents to rerank.
     * @return ranked results with original index and score.
     * @throws RerankException if the model fails.
     */
    public List<RerankResult> rerank(String query, List<String> documents) throws RerankException {
        return rerank(query, documents, 0);
    }

    /**
     * Score each document against the query and return the top-N results
     * sorted by descending relevance score.
     *
     * @param query     the search query.
     * @param documents candidate documents to rerank.
     * @param topN      maximum number of results to return;
     *                  {@code 0} means return all.
     * @return ranked results with original index and score.
     * @throws RerankException if the model fails.
     */
    public List<RerankResult> rerank(String query, List<String> documents, int topN)
            throws RerankException {
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(documents, "documents");
        return delegate.rerank(new RerankRequest(query, documents, topN));
    }

    /** Human-readable model identifier. */
    public String modelName() {
        return delegate.modelName();
    }

    /** Whether the underlying model is initialized and ready. */
    public boolean isReady() {
        return delegate.isReady();
    }

    /**
     * Return the underlying {@link Reranker} delegate.
     * <p>
     * This is intended for adapter layers (e.g. the JSON-RPC sidecar) that
     * need to bridge between the high-level runtime facade and internal
     * handler interfaces. Application code should prefer the typed methods
     * on this class instead.
     */
    public Reranker unwrapReranker() {
        return delegate;
    }

    @Override
    public void close() {
        delegate.close();
    }
}
