package com.aresstack.windirectml.encoder;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

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
     * for {@code all-MiniLM-L6-v2}, 768 for {@code e5-base}).
     */
    int dimension();

    EmbeddingVector embed(EmbeddingRequest request) throws EmbeddingException;

    /**
     * Embed a batch of requests in one logical call.
     * <p>
     * The default implementation simply iterates and calls
     * {@link #embed(EmbeddingRequest)} per item – it is therefore correct
     * for every backend but does not exploit any batching potential.
     * Backends that can dispatch multiple inputs in one GPU/SIMD pass
     * (e.g. {@code DirectMlBertEncoder}) override this method to coalesce
     * inputs by pad-bucket and run one encoder forward per bucket.
     * <p>
     * Order is preserved: {@code result.get(i)} is the embedding for
     * {@code requests.get(i)}.
     *
     * @param requests non-null, non-empty list. Individual requests must
     *                 still satisfy {@link EmbeddingRequest}'s contract
     *                 (non-blank text, etc.).
     * @return one {@link EmbeddingVector} per input, in input order.
     * @throws EmbeddingException if any underlying call fails. The batch
     *                            is treated atomically – callers must not assume partial
     *                            success.
     */
    default List<EmbeddingVector> embedBatch(List<EmbeddingRequest> requests) throws EmbeddingException {
        Objects.requireNonNull(requests, "requests");
        if (requests.isEmpty()) {
            throw new IllegalArgumentException("requests must not be empty");
        }
        List<EmbeddingVector> out = new ArrayList<>(requests.size());
        for (EmbeddingRequest r : requests) {
            out.add(embed(r));
        }
        return out;
    }
}
