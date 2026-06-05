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
}
