package com.aresstack.windirectml.runtime.facade;

import com.aresstack.windirectml.encoder.e5.E5Variant;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Configuration for loading an embedding model via
 * {@link LocalMlRuntime#loadEmbeddingModel(EmbeddingModelConfig)}.
 *
 * @param modelDir    path to the directory containing model weights
 *                    ({@code model.safetensors}) and tokenizer
 *                    ({@code tokenizer.json}).
 * @param modelFamily the encoder family ({@code "minilm"} or {@code "e5"}).
 *                    Determines which weight-loading path is used.
 * @param prefix      optional text prefix prepended to every input
 *                    (e.g. {@code "query: "} or {@code "passage: "} for E5).
 *                    {@code null} means no prefix.
 * @param e5Variant   required for E5 family: selects the WordPiece E5 variant
 *                    ({@code SMALL_V2}, {@code BASE_V2}, or {@code LARGE_V2}).
 *                    {@code null} for non-E5 families.
 */
public record EmbeddingModelConfig(Path modelDir, String modelFamily, String prefix,
                                   E5Variant e5Variant) {

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
        return new EmbeddingModelConfig(modelDir, "minilm", null, null);
    }

    /**
     * Convenience: create a config for a supported WordPiece E5 variant.
     * <p>
     * Supported variants: {@link E5Variant#SMALL_V2}, {@link E5Variant#BASE_V2},
     * {@link E5Variant#LARGE_V2}. XLM-R/SentencePiece E5 models (e.g.
     * {@code danielheinz/e5-base-sts-en-de}) are <b>not yet supported</b>.
     *
     * @param modelDir path to the model directory.
     * @param variant  the specific E5 variant to load.
     * @param prefix   E5 prefix ({@code "query: "} or {@code "passage: "}).
     */
    public static EmbeddingModelConfig e5(Path modelDir, E5Variant variant, String prefix) {
        Objects.requireNonNull(variant, "variant");
        return new EmbeddingModelConfig(modelDir, "e5", prefix, variant);
    }
}
