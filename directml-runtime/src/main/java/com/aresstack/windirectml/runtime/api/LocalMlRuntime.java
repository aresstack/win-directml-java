package com.aresstack.windirectml.runtime.api;

import com.aresstack.windirectml.encoder.EmbeddingException;

import java.util.Objects;

/** Public Java 21 entry point for local embeddings and reranking. */
public final class LocalMlRuntime {
    private final Backend backend;
    private final com.aresstack.windirectml.runtime.facade.LocalMlRuntime delegate;

    private LocalMlRuntime(Builder builder) {
        this.backend = Objects.requireNonNull(builder.backend, "backend");
        var config = com.aresstack.windirectml.runtime.facade.LocalMlRuntimeConfig.builder()
                .backend(backend.toFacade())
                .build();
        this.delegate = com.aresstack.windirectml.runtime.facade.LocalMlRuntime.create(config);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static LocalMlRuntime create() {
        return builder().build();
    }

    public Backend backend() {
        return backend;
    }

    public EmbeddingModelHandle loadEmbeddings(EmbeddingConfig config) throws EmbeddingException {
        Objects.requireNonNull(config, "config");
        return new EmbeddingModelHandle(delegate.loadEmbeddingModel(config.toFacade()), backend);
    }

    public RerankerModelHandle loadReranker(RerankerConfig config) throws EmbeddingException {
        Objects.requireNonNull(config, "config");
        return new RerankerModelHandle(delegate.loadRerankerModel(config.toFacade()), backend);
    }

    public static final class Builder {
        private Backend backend = Backend.AUTO;

        private Builder() { }

        public Builder backend(Backend backend) {
            this.backend = Objects.requireNonNull(backend, "backend");
            return this;
        }

        public LocalMlRuntime build() {
            return new LocalMlRuntime(this);
        }
    }
}
