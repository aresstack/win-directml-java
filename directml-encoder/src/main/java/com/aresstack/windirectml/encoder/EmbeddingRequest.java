package com.aresstack.windirectml.encoder;

import java.util.Objects;

/**
 * Request to an {@link EmbeddingModel}.
 *
 * @param text     non-blank text to embed.
 * @param normalize whether to L2-normalize the resulting vector (default {@code true}).
 * @param prefix   optional input prefix (e.g. {@code "query: "} or
 *                 {@code "passage: "} for the E5 family); {@code null} = no prefix.
 */
public record EmbeddingRequest(String text, boolean normalize, String prefix) {

    public EmbeddingRequest {
        Objects.requireNonNull(text, "text");
        if (text.isBlank()) {
            throw new IllegalArgumentException("text must not be blank");
        }
    }

    public static EmbeddingRequest of(String text) {
        return new EmbeddingRequest(text, true, null);
    }
}

