package com.aresstack.windirectml.runtime.facade;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * High-level embedding model handle returned by
 * {@link LocalMlRuntime#loadEmbeddingModel(EmbeddingModelConfig)}.
 * <p>
 * Wraps the internal {@link EmbeddingModel} and provides a simple
 * string-in / float[]-out API without requiring callers to deal with
 * {@link EmbeddingRequest} or {@link EmbeddingVector} records.
 * <p>
 * Instances are thread-safe if the underlying model is thread-safe
 * (the shipped CPU and DirectML backends are).
 */
public final class LocalEmbeddingModel implements AutoCloseable {

    private final EmbeddingModel delegate;
    private final String prefix;

    LocalEmbeddingModel(EmbeddingModel delegate, String prefix) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.prefix = prefix;
    }

    /**
     * Produce an embedding vector for a single text input.
     *
     * @param text non-blank text to embed.
     * @return the embedding vector as a float array.
     * @throws EmbeddingException if the model fails.
     */
    public float[] embed(String text) throws EmbeddingException {
        Objects.requireNonNull(text, "text");
        EmbeddingRequest req = new EmbeddingRequest(text, true, prefix);
        return delegate.embed(req).values();
    }

    /**
     * Produce embedding vectors for a batch of texts.
     * <p>
     * Order is preserved: {@code result.get(i)} corresponds to
     * {@code texts.get(i)}.
     *
     * @param texts non-null, non-empty list of texts.
     * @return list of embedding vectors in input order.
     * @throws EmbeddingException if the model fails.
     */
    public List<float[]> embedBatch(List<String> texts) throws EmbeddingException {
        Objects.requireNonNull(texts, "texts");
        if (texts.isEmpty()) {
            throw new IllegalArgumentException("texts must not be empty");
        }
        List<EmbeddingRequest> requests = new ArrayList<>(texts.size());
        for (String t : texts) {
            requests.add(new EmbeddingRequest(t, true, prefix));
        }
        List<EmbeddingVector> vectors = delegate.embedBatch(requests);
        List<float[]> result = new ArrayList<>(vectors.size());
        for (EmbeddingVector v : vectors) {
            result.add(v.values());
        }
        return result;
    }

    /** The output dimension of the loaded model (e.g. 384 for MiniLM). */
    public int dimension() {
        return delegate.dimension();
    }

    /** Whether the underlying model is initialized and ready. */
    public boolean isReady() {
        return delegate.isReady();
    }

    /**
     * Return the underlying {@link EmbeddingModel} delegate.
     * <p>
     * This is intended for adapter layers (e.g. the JSON-RPC sidecar) that
     * need to bridge between the high-level runtime facade and internal
     * handler interfaces. Application code should prefer the typed methods
     * on this class instead.
     */
    public EmbeddingModel unwrapModel() {
        return delegate;
    }

    @Override
    public void close() {
        if (delegate instanceof AutoCloseable ac) {
            try {
                ac.close();
            } catch (Exception e) {
                throw new RuntimeException("Failed to close embedding model", e);
            }
        }
    }
}
