package com.aresstack.windirectml.encoder;

import java.nio.file.Path;

/**
 * Describes an encoder architecture family (MiniLM, E5, JinaBERT, …).
 * <p>
 * Implementations declare the structural choices needed to drive the
 * shared {@link EncoderRuntime}:
 * <ul>
 *   <li>tensor names and shapes expected in the safetensors file,</li>
 *   <li>position-embedding flavor (absolute, RoPE),</li>
 *   <li>activation (GELU, SwiGLU),</li>
 *   <li>normalization (LayerNorm, RMSNorm),</li>
 *   <li>{@link PoolingStrategy} for sentence vectors,</li>
 *   <li>whether the output is L2-normalized by default.</li>
 * </ul>
 */
public interface EncoderArchitecture {

    String name();

    int hiddenSize();

    int numLayers();

    int numAttentionHeads();

    int maxSequenceLength();

    int outputDimension();

    PoolingStrategy poolingStrategy();

    boolean l2NormalizeByDefault();

    /**
     * Load weights for this architecture from a model directory.
     * <p>
     * The default model layout is expected to contain at least
     * {@code config.json}, {@code tokenizer.json} and
     * {@code model.safetensors}.
     */
    EncoderWeights loadWeights(Path modelDir) throws EmbeddingException;
}

