package com.aresstack.windirectml.encoder;

/**
 * Pooling strategies used by sentence-transformer style encoders.
 * <p>
 * The strategy collapses the {@code [seq_len, hidden]} token-embedding
 * matrix into a single {@code [hidden]} sentence vector.
 */
public enum PoolingStrategy {

    /**
     * Mean over token embeddings, weighted by the attention mask.
     */
    MEAN,

    /**
     * Use the embedding of the {@code [CLS]} token.
     */
    CLS,

    /**
     * Elementwise maximum across tokens (mask-aware).
     */
    MAX
}

