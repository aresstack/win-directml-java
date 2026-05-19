package com.aresstack.windirectml.encoder.reranker;

import java.util.List;
import java.util.Objects;

/**
 * Cross-encoder reranking request.
 *
 * @param query     the search query, embedded once per candidate pair.
 * @param documents candidate documents (typically the recall top-N from
 *                  a bi-encoder retrieval).
 * @param topN      number of best-scoring documents to return; values
 *                  {@code <= 0} or {@code >= documents.size()} mean
 *                  "return all, sorted".
 */
public record RerankRequest(String query, List<String> documents, int topN) {

    public RerankRequest {
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(documents, "documents");
        if (query.isBlank()) throw new IllegalArgumentException("query must not be blank");
        if (documents.isEmpty()) throw new IllegalArgumentException("documents must not be empty");
        documents = List.copyOf(documents);
    }

    /** Effective top-N: never larger than {@link #documents()} size. */
    public int effectiveTopN() {
        if (topN <= 0 || topN > documents.size()) return documents.size();
        return topN;
    }
}

