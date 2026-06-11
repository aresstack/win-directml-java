package com.aresstack.windirectml.inference.decoderonly;

/**
 * Minimal shape contract for decoder-only causal language models.
 *
 * <p>The contract deliberately contains only architecture dimensions that are
 * shared by Qwen-like and Llama-like runtimes. Family-specific configuration
 * stays in the concrete model packages.</p>
 */
public interface DecoderOnlyConfig {

    int numHiddenLayers();

    int hiddenSize();

    int intermediateSize();

    int numAttentionHeads();

    int numKeyValueHeads();

    int headDim();

    int maxPositionEmbeddings();

    int vocabSize();

    float rmsNormEps();

    float ropeTheta();

    /**
     * Build a plain immutable {@link DecoderOnlyConfig} from explicit dimensions. This is the adapter seam for model
     * families whose own config type cannot implement this interface directly (e.g. records whose components have
     * incompatible types). The argument order matches the accessor declaration order above.
     */
    static DecoderOnlyConfig of(int numHiddenLayers,
                                int hiddenSize,
                                int intermediateSize,
                                int numAttentionHeads,
                                int numKeyValueHeads,
                                int headDim,
                                int maxPositionEmbeddings,
                                int vocabSize,
                                float rmsNormEps,
                                float ropeTheta) {
        return new DecoderOnlyConfigValues(numHiddenLayers, hiddenSize, intermediateSize, numAttentionHeads,
                numKeyValueHeads, headDim, maxPositionEmbeddings, vocabSize, rmsNormEps, ropeTheta);
    }
}
