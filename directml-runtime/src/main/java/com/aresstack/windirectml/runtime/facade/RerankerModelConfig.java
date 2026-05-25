package com.aresstack.windirectml.runtime.facade;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Configuration for loading a reranker model via
 * {@link LocalMlRuntime#loadRerankerModel(RerankerModelConfig)}.
 *
 * @param modelDir path to the directory containing cross-encoder weights
 *                 and tokenizer.
 */
public record RerankerModelConfig(Path modelDir) {

    public RerankerModelConfig {
        Objects.requireNonNull(modelDir, "modelDir");
    }
}
