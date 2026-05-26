package com.aresstack.windirectml.runtime.api;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Public configuration for loading an embedding model.
 */
public final class EmbeddingConfig {
    private final EmbeddingModelId model;
    private final Path modelDir;
    private final String prefix;

    private EmbeddingConfig(Builder builder) {
        this.model = Objects.requireNonNull(builder.model, "model");
        this.modelDir = Objects.requireNonNull(builder.modelDir, "modelDir");
        this.prefix = builder.prefixSet ? builder.prefix : model.defaultPrefix();
    }

    public static Builder builder() {
        return new Builder();
    }

    public EmbeddingModelId model() {
        return model;
    }

    public Path modelDir() {
        return modelDir;
    }

    public String prefix() {
        return prefix;
    }

    com.aresstack.windirectml.runtime.facade.EmbeddingModelConfig toFacade() {
        if (model.e5Variant() != null) {
            return com.aresstack.windirectml.runtime.facade.EmbeddingModelConfig.e5(
                    modelDir, model.e5Variant(), prefix);
        }
        return new com.aresstack.windirectml.runtime.facade.EmbeddingModelConfig(
                modelDir, model.family(), prefix, null);
    }

    public static final class Builder {
        private EmbeddingModelId model;
        private Path modelDir;
        private String prefix;
        private boolean prefixSet;

        private Builder() { }

        public Builder model(EmbeddingModelId model) {
            this.model = Objects.requireNonNull(model, "model");
            return this;
        }

        public Builder modelDir(Path modelDir) {
            this.modelDir = Objects.requireNonNull(modelDir, "modelDir");
            return this;
        }

        public Builder prefix(String prefix) {
            this.prefix = prefix;
            this.prefixSet = true;
            return this;
        }

        public EmbeddingConfig build() {
            return new EmbeddingConfig(this);
        }
    }
}
