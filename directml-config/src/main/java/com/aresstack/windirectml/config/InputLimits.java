package com.aresstack.windirectml.config;

/**
 * Defines the input-size limits enforced by the sidecar and validated locally
 * by the Java-8 client. These constants protect against oversized requests
 * that could cause out-of-memory errors or unacceptable latency.
 *
 * <p>All limits are <b>inclusive</b> upper bounds. A request that exceeds any
 * limit is rejected with error code {@code -32007 LIMIT_EXCEEDED} on the
 * sidecar side, or with a local {@code SidecarException} on the client side.
 *
 * <p>The values are designed for BERT-style encoders with a 512-token context
 * window and typical RAG / reranking workloads on consumer-grade GPUs.
 * They can be overridden via system properties (documented in PROTOCOL.md).
 *
 * <p>Java-8 compatible.
 */
public final class InputLimits {

    private InputLimits() {}

    // ── Text length ─────────────────────────────────────────────────────

    /**
     * Maximum character length for a single text input (embed, embedBatch entry,
     * summarize). 32 768 characters ≈ ~8 000 tokens; well above a 512-token
     * BERT context but catches accidental megabyte blobs.
     */
    public static final int MAX_TEXT_LENGTH = 32_768;

    /** System property to override {@link #MAX_TEXT_LENGTH}. */
    public static final String PROP_MAX_TEXT_LENGTH = "windirectml.limits.maxTextLength";

    // ── embedBatch ──────────────────────────────────────────────────────

    /**
     * Maximum number of texts in a single {@code embedBatch} call.
     * 256 texts × 32 KB worst-case ≈ 8 MB – comfortable for 8 GB GPU memory.
     */
    public static final int MAX_EMBED_BATCH_SIZE = 256;

    /** System property to override {@link #MAX_EMBED_BATCH_SIZE}. */
    public static final String PROP_MAX_EMBED_BATCH_SIZE = "windirectml.limits.maxEmbedBatchSize";

    // ── rerank ──────────────────────────────────────────────────────────

    /**
     * Maximum number of documents in a single {@code rerank} call.
     * Cross-encoder forward passes are O(n); 256 pairs is already expensive
     * on CPU.
     */
    public static final int MAX_RERANK_DOCUMENTS = 256;

    /** System property to override {@link #MAX_RERANK_DOCUMENTS}. */
    public static final String PROP_MAX_RERANK_DOCUMENTS = "windirectml.limits.maxRerankDocuments";

    /**
     * Maximum character length for a single document in {@code rerank}.
     * Same rationale as {@link #MAX_TEXT_LENGTH}.
     */
    public static final int MAX_RERANK_DOCUMENT_LENGTH = 32_768;

    /** System property to override {@link #MAX_RERANK_DOCUMENT_LENGTH}. */
    public static final String PROP_MAX_RERANK_DOCUMENT_LENGTH = "windirectml.limits.maxRerankDocumentLength";

    // ── Resolved values (honour system-property overrides) ──────────────

    /**
     * Returns the effective max text length, honouring the system property
     * override if set to a positive integer.
     */
    public static int maxTextLength() {
        return resolveInt(PROP_MAX_TEXT_LENGTH, MAX_TEXT_LENGTH);
    }

    /** Returns the effective max embed-batch size. */
    public static int maxEmbedBatchSize() {
        return resolveInt(PROP_MAX_EMBED_BATCH_SIZE, MAX_EMBED_BATCH_SIZE);
    }

    /** Returns the effective max rerank document count. */
    public static int maxRerankDocuments() {
        return resolveInt(PROP_MAX_RERANK_DOCUMENTS, MAX_RERANK_DOCUMENTS);
    }

    /** Returns the effective max rerank document length. */
    public static int maxRerankDocumentLength() {
        return resolveInt(PROP_MAX_RERANK_DOCUMENT_LENGTH, MAX_RERANK_DOCUMENT_LENGTH);
    }

    // ── internal ────────────────────────────────────────────────────────

    private static int resolveInt(String prop, int defaultValue) {
        String v = System.getProperty(prop);
        if (v != null && !v.isEmpty()) {
            try {
                int parsed = Integer.parseInt(v.trim());
                if (parsed > 0) return parsed;
            } catch (NumberFormatException ignored) {
                // fall through to default
            }
        }
        return defaultValue;
    }
}
