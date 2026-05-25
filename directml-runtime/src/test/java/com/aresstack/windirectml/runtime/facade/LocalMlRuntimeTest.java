package com.aresstack.windirectml.runtime.facade;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link LocalMlRuntime} facade creation and unsupported-model
 * error behaviour. These tests do NOT load real model weights – they
 * verify the structural behaviour of the facade.
 */
class LocalMlRuntimeTest {

    @Test
    void createWithDefaultsUsesAutoBackend() {
        LocalMlRuntime runtime = LocalMlRuntime.create();
        assertEquals(Backend.AUTO, runtime.backend());
    }

    @Test
    void createWithExplicitBackend() {
        var config = LocalMlRuntimeConfig.builder().backend(Backend.CPU).build();
        LocalMlRuntime runtime = LocalMlRuntime.create(config);
        assertEquals(Backend.CPU, runtime.backend());
    }

    @Test
    void loadEmbeddingModelRejectsUnsupportedFamily() {
        LocalMlRuntime runtime = LocalMlRuntime.create();
        var cfg = new EmbeddingModelConfig(Path.of("dummy"), "xlm-r", null);
        var ex = assertThrows(UnsupportedModelException.class,
                () -> runtime.loadEmbeddingModel(cfg));
        assertEquals("xlm-r", ex.family());
        assertNotNull(ex.getMessage());
    }

    @Test
    void loadEmbeddingModelRejectsUnknownFamily() {
        LocalMlRuntime runtime = LocalMlRuntime.create();
        var cfg = new EmbeddingModelConfig(Path.of("dummy"), "totally-unknown", null);
        var ex = assertThrows(UnsupportedModelException.class,
                () -> runtime.loadEmbeddingModel(cfg));
        assertEquals("totally-unknown", ex.family());
        assertTrue(ex.getMessage().contains("not recognized"));
    }

    @Test
    void embeddingModelConfigMiniLmFactory() {
        var cfg = EmbeddingModelConfig.miniLm(Path.of("model/all-MiniLM-L6-v2"));
        assertEquals("minilm", cfg.modelFamily());
        assertNull(cfg.prefix());
        assertEquals(Path.of("model/all-MiniLM-L6-v2"), cfg.modelDir());
    }

    @Test
    void embeddingModelConfigE5Factory() {
        var cfg = EmbeddingModelConfig.e5(Path.of("model/e5"), "query: ");
        assertEquals("e5", cfg.modelFamily());
        assertEquals("query: ", cfg.prefix());
    }

    @Test
    void embeddingModelConfigRejectsBlankFamily() {
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingModelConfig(Path.of("x"), " ", null));
    }

    @Test
    void backendParsing() {
        assertEquals(Backend.AUTO, Backend.parse(null));
        assertEquals(Backend.AUTO, Backend.parse(""));
        assertEquals(Backend.AUTO, Backend.parse("auto"));
        assertEquals(Backend.CPU, Backend.parse("cpu"));
        assertEquals(Backend.DIRECTML, Backend.parse("directml"));
        assertEquals(Backend.DIRECTML, Backend.parse("dml"));
        assertThrows(IllegalArgumentException.class, () -> Backend.parse("unknown"));
    }

    @Test
    void unsupportedModelExceptionCarriesFamily() {
        var ex = new UnsupportedModelException("test-family", "not supported");
        assertEquals("test-family", ex.family());
        assertEquals("not supported", ex.getMessage());
    }
}
