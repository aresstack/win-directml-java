package com.aresstack.windirectml.runtime;

/**
 * Supported embedding model families.
 *
 * @see WinDirectMlRuntime.Builder#embeddingFamily(EmbeddingFamily)
 */
public enum EmbeddingFamily {

    /** {@code sentence-transformers/all-MiniLM-L6-v2} (384-dim). */
    MINILM,

    /** {@code intfloat/multilingual-e5-*} family (768-dim for base). */
    E5
}
