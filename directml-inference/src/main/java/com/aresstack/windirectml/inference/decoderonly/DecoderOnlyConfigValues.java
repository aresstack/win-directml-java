package com.aresstack.windirectml.inference.decoderonly;

/**
 * Plain immutable {@link DecoderOnlyConfig} carrier.
 *
 * <p>Used by {@link DecoderOnlyConfig#of} so model families can expose a family-neutral shape view without their own
 * config type having to implement the interface. Carries no behaviour beyond the accessors.</p>
 */
record DecoderOnlyConfigValues(
        int numHiddenLayers,
        int hiddenSize,
        int intermediateSize,
        int numAttentionHeads,
        int numKeyValueHeads,
        int headDim,
        int maxPositionEmbeddings,
        int vocabSize,
        float rmsNormEps,
        float ropeTheta
) implements DecoderOnlyConfig {
}
