package com.aresstack.windirectml.runtime.api;

import com.aresstack.windirectml.encoder.reranker.RerankException;

import java.util.List;
import java.util.Objects;

/**
 * Public reranker model handle for Java 21 applications.
 */
public final class RerankerModelHandle implements AutoCloseable {
    private final com.aresstack.windirectml.runtime.facade.LocalRerankerModel delegate;
    private final Backend selectedBackend;

    RerankerModelHandle(com.aresstack.windirectml.runtime.facade.LocalRerankerModel delegate,
                        Backend selectedBackend) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.selectedBackend = Objects.requireNonNull(selectedBackend, "selectedBackend");
    }

    public List<RerankResult> rerank(String query, List<String> documents) throws RerankException {
        return rerank(query, documents, 0);
    }

    public List<RerankResult> rerank(String query, List<String> documents, int topN) throws RerankException {
        return delegate.rerank(query, documents, topN).stream()
                .map(RerankResult::fromInternal)
                .toList();
    }

    public String modelName() {
        return delegate.modelName();
    }

    public boolean isReady() {
        return delegate.isReady();
    }

    /**
     * Backend selected for this loaded model. For runtime backend AUTO this is
     * currently the requested AUTO value until provider selection is made
     * observable by the lower-level facade.
     */
    public Backend selectedBackend() {
        return selectedBackend;
    }

    @Override
    public void close() {
        delegate.close();
    }
}
