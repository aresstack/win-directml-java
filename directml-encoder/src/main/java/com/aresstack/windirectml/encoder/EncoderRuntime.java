package com.aresstack.windirectml.encoder;

/**
 * Skeleton interface for the shared encoder runtime.
 * <p>
 * Bridges the architecture-neutral {@link EmbeddingModel} surface with
 * the DirectML / D3D12 layer (still to be extracted in Milestone 4).
 * Concrete implementations (MiniLM, E5, JinaBERT) will plug in via an
 * {@link EncoderArchitecture} + {@link EncoderWeights} pair.
 */
public interface EncoderRuntime extends AutoCloseable {

    void initialize(EncoderArchitecture architecture,
                    EncoderWeights weights,
                    EncoderTokenizer tokenizer) throws EmbeddingException;

    boolean isReady();

    int dimension();

    EmbeddingVector embed(EmbeddingRequest request) throws EmbeddingException;

    @Override
    void close();
}

