package com.aresstack.windirectml.runtime.facade;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Configuration for loading an embedding model via
 * {@link LocalMlRuntime#loadEmbeddingModel(EmbeddingModelConfig)}.
 *
 * @param modelDir  path to the directory containing model weights
 *                  ({@code model.safetensors}) and tokenizer
 *                  ({@code tokenizer.json}).
 * @param modelFamily the encoder family ({@code "minilm"} or {@code "e5"}).
 *                    Determines which weight-loading path is used.
 * @param prefix    optional text prefix prepended to every input
 *                  (e.g. {@code "query: "} or {@code "passage: "} for E5).
 *                  {@code null} means no prefix.
 */
public record EmbeddingModelConfig(Path modelDir, String modelFamily, String prefix) {

    public EmbeddingModelConfig {
        Objects.requireNonNull(modelDir, "modelDir");
        Objects.requireNonNull(modelFamily, "modelFamily");
        if (modelFamily.isBlank()) {
            throw new IllegalArgumentException("modelFamily must not be blank");
        }
    }

    /**
     * Convenience: create a config for MiniLM (the default model family).
     *
     * @param modelDir path to the model directory.
     */
    public static EmbeddingModelConfig miniLm(Path modelDir) {
        return new EmbeddingModelConfig(modelDir, "minilm", null);
    }

    /**
     * Convenience: create a config for an E5 model.
     *
     * @param modelDir path to the model directory.
     * @param prefix   E5 prefix ({@code "query: "} or {@code "passage: "}).
     */
    public static EmbeddingModelConfig e5(Path modelDir, String prefix) {
        return new EmbeddingModelConfig(modelDir, "e5", prefix);
    }
}
