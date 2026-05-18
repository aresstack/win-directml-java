package com.aresstack.windirectml.encoder;

/**
 * Public use-case API for sentence/text embeddings.
 * <p>
 * The encoder runtime is intentionally separate from the decoder runtime
 * ({@link com.aresstack.windirectml.inference.Phi3InferenceEngine Phi-3}).
 * Embeddings have a different lifecycle: no KV cache, no chat template,
 * no streaming – just {@code text → fixed-size vector}.
 * <p>
 * Implementations may target different encoder families (MiniLM, E5,
 * JinaBERT, …) but always expose this interface to the sidecar.
 */
public interface EmbeddingModel {

    /**
     * @return {@code true} once the encoder is initialized and ready.
     */
    boolean isReady();

    /**
     * @return the output dimensionality of the produced vectors (e.g. 384
     *         for {@code all-MiniLM-L6-v2}, 768 for {@code e5-base}).
     */
    int dimension();

    EmbeddingVector embed(EmbeddingRequest request) throws EmbeddingException;
}

