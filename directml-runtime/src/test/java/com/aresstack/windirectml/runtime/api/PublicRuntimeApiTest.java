package com.aresstack.windirectml.runtime.api;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class PublicRuntimeApiTest {

    @Test
    void embeddingConfigUsesStrongModelId() {
        var config = EmbeddingConfig.builder()
                .model(EmbeddingModelId.MINILM_L6_V2)
                .modelDir(Path.of("model/all-MiniLM-L6-v2"))
                .build();

        assertEquals(EmbeddingModelId.MINILM_L6_V2, config.model());
        assertEquals(Path.of("model/all-MiniLM-L6-v2"), config.modelDir());
    }

    @Test
    void e5ConfigAppliesDefaultPrefix() {
        var config = EmbeddingConfig.builder()
                .model(EmbeddingModelId.E5_BASE_V2)
                .modelDir(Path.of("model/e5-base-v2"))
                .build();

        assertEquals("query: ", config.prefix());
    }

    @Test
    void rerankerConfigUsesStrongModelId() {
        var config = RerankerConfig.builder()
                .model(RerankerModelId.MS_MARCO_MINILM_L6)
                .modelDir(Path.of("model/cross-encoder-ms-marco-MiniLM-L-6-v2"))
                .build();

        assertEquals(RerankerModelId.MS_MARCO_MINILM_L6, config.model());
        assertEquals(Path.of("model/cross-encoder-ms-marco-MiniLM-L-6-v2"), config.modelDir());
    }

    @Test
    void rerankerConfigDerivesModelDirFromModelId() {
        var config = RerankerConfig.builder()
                .model(RerankerModelId.MS_MARCO_MINILM_L6)
                .build();

        // modelDir derived from model.directoryName() under "model/"
        assertEquals(
                Path.of("model", RerankerModelId.MS_MARCO_MINILM_L6.directoryName()),
                config.modelDir());
    }

    @Test
    void rerankerConfigRejectsInconsistentModelDir() {
        assertThrows(IllegalArgumentException.class, () ->
                RerankerConfig.builder()
                        .model(RerankerModelId.MS_MARCO_MINILM_L6)
                        .modelDir(Path.of("model/bge-reranker-base"))   // wrong name for chosen model
                        .build());
    }

    @Test
    void runtimeBuilderDefaultsToAuto() {
        var runtime = MlRuntime.create();
        assertEquals(Backend.AUTO, runtime.backend());
    }

    @Test
    void runtimeBuilderAcceptsExplicitBackend() {
        var runtime = MlRuntime.builder()
                .backend(Backend.CPU)
                .build();
        assertEquals(Backend.CPU, runtime.backend());
    }

    @Test
    void publicRerankResultValidatesIndex() {
        assertThrows(IllegalArgumentException.class, () -> new RerankResult(-1, 0.5));
        assertEquals(0.5, new RerankResult(0, 0.5).score());
    }
}
