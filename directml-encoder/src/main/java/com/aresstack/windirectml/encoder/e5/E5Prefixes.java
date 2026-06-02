package com.aresstack.windirectml.encoder.e5;

import com.aresstack.windirectml.encoder.EmbeddingRequest;

/**
 * Conventional input prefixes the E5 family expects. E5 models are
 * fine-tuned so that retrieval queries and corpus passages must be
 * prefixed before tokenisation:
 * <pre>
 *   query:   "query: "   + text
 *   passage: "passage: " + text
 * </pre>
 * The prefix is glued onto the raw text at encoding time via
 * {@link EmbeddingRequest#prefix()}, so the rest of the pipeline
 * (tokenizer, encoder, pooling) remains model-agnostic.
 * <p>
 * Use {@link #request(String, Role, boolean)} for a one-liner that
 * builds an {@link EmbeddingRequest} with the matching prefix.
 */
public final class E5Prefixes {

    public static final String QUERY = "query: ";
    public static final String PASSAGE = "passage: ";

    public enum Role {
        QUERY,
        PASSAGE;

        public String prefix() {
            return this == QUERY ? E5Prefixes.QUERY : E5Prefixes.PASSAGE;
        }
    }

    private E5Prefixes() {
    }

    /**
     * Build an {@link EmbeddingRequest} with the E5-appropriate prefix.
     */
    public static EmbeddingRequest request(String text, Role role, boolean normalize) {
        return new EmbeddingRequest(text, normalize, role.prefix());
    }
}

