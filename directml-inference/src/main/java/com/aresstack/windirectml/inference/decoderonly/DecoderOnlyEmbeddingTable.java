package com.aresstack.windirectml.inference.decoderonly;

/**
 * Token-embedding lookup seam for the shared decoder-only WARP forward pass.
 *
 * <p>The forward pass only ever needs to copy one embedding row into a reusable buffer; the actual storage (dense
 * float view, tied/untied embedding, …) stays in the model family. This keeps the generic runtime independent of any
 * family-specific tensor type.</p>
 */
@FunctionalInterface
public interface DecoderOnlyEmbeddingTable {

    /**
     * Copy the embedding row for {@code tokenId} into {@code target[0..hiddenSize)}.
     *
     * @param tokenId the token id whose embedding row is requested
     * @param target  destination buffer, at least {@code hiddenSize} long
     */
    void copyRowInto(int tokenId, float[] target);
}
