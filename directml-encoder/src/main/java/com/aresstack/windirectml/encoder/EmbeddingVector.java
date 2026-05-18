package com.aresstack.windirectml.encoder;

import java.util.Objects;

/**
 * A dense embedding vector produced by an {@link EmbeddingModel}.
 *
 * @param values    raw float values (length = {@link #dimension()}).
 * @param dimension number of components (kept explicit for protocol use).
 * @param model     identifier of the model that produced the vector,
 *                  e.g. {@code "sentence-transformers/all-MiniLM-L6-v2"}.
 * @param normalized whether the values are L2-normalized.
 */
public record EmbeddingVector(float[] values, int dimension, String model, boolean normalized) {

    public EmbeddingVector {
        Objects.requireNonNull(values, "values");
        Objects.requireNonNull(model, "model");
        if (values.length != dimension) {
            throw new IllegalArgumentException(
                    "values.length=" + values.length + " does not match dimension=" + dimension);
        }
    }
}

