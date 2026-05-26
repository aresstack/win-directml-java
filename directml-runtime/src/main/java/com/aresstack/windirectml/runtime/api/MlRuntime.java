package com.aresstack.windirectml.runtime.api;

import com.aresstack.windirectml.encoder.EmbeddingException;

import java.util.Objects;

/**
 * Public Java 21 entry point for local embeddings and reranking.
 *
 * <p>This is the primary entry point for the {@code runtime.api} package.
 * Use this class to load embedding and reranker models without starting
 * the JSON-RPC sidecar process.
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * MlRuntime runtime = MlRuntime.create();
 *
 * // Embeddings
 * var embedCfg = EmbeddingConfig.builder()
 *         .model(EmbeddingModelId.E5_BASE_V2)
 *         .modelDir(Path.of("model/e5-base-v2"))
 *         .build();
 * try (EmbeddingModelHandle embeddings = runtime.loadEmbeddings(embedCfg)) {
 *     float[] vector = embeddings.embed("hello world");
 * }
 *
 * // Reranking
 * var rerankCfg = RerankerConfig.builder()
 *         .model(RerankerModelId.MS_MARCO_MINILM_L6)
 *         .build();
 * try (RerankerModelHandle reranker = runtime.loadReranker(rerankCfg)) {
 *     var results = reranker.rerank("search query", List.of("doc1", "doc2"));
 * }
 * }</pre>
 *
 * <p>Consumers of the lower-level {@code runtime.facade} package may migrate
 * to this class at their convenience. The facade's
 * {@link com.aresstack.windirectml.runtime.facade.LocalMlRuntime} is still
 * fully functional but has been deprecated in favour of this type.
 */
public final class MlRuntime {
    private final Backend backend;
    private final com.aresstack.windirectml.runtime.facade.LocalMlRuntime delegate;

    private MlRuntime(Builder builder) {
        this.backend = Objects.requireNonNull(builder.backend, "backend");
        var config = com.aresstack.windirectml.runtime.facade.LocalMlRuntimeConfig.builder()
                .backend(backend.toFacade())
                .build();
        this.delegate = com.aresstack.windirectml.runtime.facade.LocalMlRuntime.create(config);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static MlRuntime create() {
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

        public MlRuntime build() {
            return new MlRuntime(this);
        }
    }
}
