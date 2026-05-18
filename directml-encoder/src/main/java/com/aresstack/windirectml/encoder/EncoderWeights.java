package com.aresstack.windirectml.encoder;

/**
 * Marker interface for an architecture-specific weight bundle.
 * <p>
 * Concrete implementations will hold buffers (memory-mapped safetensor
 * regions) for embeddings, attention projections, MLPs, LayerNorms and
 * pooling. The shared {@link EncoderRuntime} consumes these via the
 * architecture descriptor without leaking architecture details.
 */
public interface EncoderWeights extends AutoCloseable {

    EncoderArchitecture architecture();

    @Override
    void close();
}

