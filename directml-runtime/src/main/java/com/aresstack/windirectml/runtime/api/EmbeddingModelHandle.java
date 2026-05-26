package com.aresstack.windirectml.runtime.api;

import com.aresstack.windirectml.encoder.EmbeddingException;

import java.util.List;
import java.util.Objects;

/**
 * Public embedding model handle for Java 21 applications.
 */
public final class EmbeddingModelHandle implements AutoCloseable {
    private final com.aresstack.windirectml.runtime.facade.LocalEmbeddingModel delegate;
    private final Backend selectedBackend;

    EmbeddingModelHandle(com.aresstack.windirectml.runtime.facade.LocalEmbeddingModel delegate,
                         Backend selectedBackend) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.selectedBackend = Objects.requireNonNull(selectedBackend, "selectedBackend");
    }

    public float[] embed(String text) throws EmbeddingException {
        return delegate.embed(text);
    }

    public List<float[]> embedBatch(List<String> texts) throws EmbeddingException {
        return delegate.embedBatch(texts);
    }

    public int dimension() {
        return delegate.dimension();
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
