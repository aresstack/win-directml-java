package com.aresstack.windirectml.encoder.reranker;

/**
 * A single reranked entry. {@link #originalIndex()} refers to the
 * position inside the {@link RerankRequest#documents()} list as
 * provided by the caller; the host re-resolves the document text from
 * its own list (the reranker does not echo it back, to keep payloads
 * small).
 *
 * @param originalIndex 0-based index into the original documents list.
 * @param score         raw classifier logit (higher = more relevant).
 *                      Cross-encoder logits are not normalised – do not
 *                      compare across different reranker models.
 */
public record RerankResult(int originalIndex, double score) {
    public RerankResult {
        if (originalIndex < 0) throw new IllegalArgumentException("originalIndex < 0");
    }
}

