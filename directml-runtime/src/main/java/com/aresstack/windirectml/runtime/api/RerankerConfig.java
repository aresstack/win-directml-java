package com.aresstack.windirectml.runtime.api;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Public configuration for loading a reranker model.
 */
public final class RerankerConfig {
    private final RerankerModelId model;
    private final Path modelDir;

    private RerankerConfig(Builder builder) {
        this.model = Objects.requireNonNull(builder.model, "model");
        this.modelDir = Objects.requireNonNull(builder.modelDir, "modelDir");
    }

    public static Builder builder() {
        return new Builder();
    }

    public RerankerModelId model() {
        return model;
    }

    public Path modelDir() {
        return modelDir;
    }

    com.aresstack.windirectml.runtime.facade.RerankerModelConfig toFacade() {
        return new com.aresstack.windirectml.runtime.facade.RerankerModelConfig(modelDir);
    }

    public static final class Builder {
        private RerankerModelId model;
        private Path modelDir;

        private Builder() { }

        public Builder model(RerankerModelId model) {
            this.model = Objects.requireNonNull(model, "model");
            return this;
        }

        public Builder modelDir(Path modelDir) {
            this.modelDir = Objects.requireNonNull(modelDir, "modelDir");
            return this;
        }

        public RerankerConfig build() {
            return new RerankerConfig(this);
        }
    }
}
