package com.aresstack.windirectml.inference.qwen;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link Qwen2Config}.
 *
 * <p>Tests config loading from JSON and derived properties.
 * No real model weights needed.
 */
class Qwen2ConfigTest {

    /**
     * Qwen2.5-Coder-0.5B config values.
     */
    private static final String CONFIG_JSON_0_5B = """
            {
              "architectures": ["Qwen2ForCausalLM"],
              "hidden_size": 896,
              "num_attention_heads": 14,
              "num_hidden_layers": 24,
              "num_key_value_heads": 2,
              "vocab_size": 151936,
              "max_position_embeddings": 32768,
              "intermediate_size": 4864,
              "rms_norm_eps": 1e-6,
              "rope_theta": 1000000.0,
              "model_type": "qwen2"
            }
            """;

    /**
     * Qwen2.5-Coder-1.5B config values (for scalability check).
     */
    private static final String CONFIG_JSON_1_5B = """
            {
              "hidden_size": 1536,
              "num_attention_heads": 12,
              "num_hidden_layers": 28,
              "num_key_value_heads": 2,
              "vocab_size": 151936,
              "max_position_embeddings": 32768,
              "intermediate_size": 8960,
              "rms_norm_eps": 1e-6,
              "rope_theta": 1000000.0
            }
            """;

    @Test
    void loadsCorrectly(@TempDir Path tmp) throws Exception {
        Path configJson = tmp.resolve("config.json");
        Files.writeString(configJson, CONFIG_JSON_0_5B);

        Qwen2Config config = Qwen2Config.load(configJson);

        assertEquals(896, config.hiddenSize());
        assertEquals(14, config.numAttentionHeads());
        assertEquals(24, config.numHiddenLayers());
        assertEquals(2, config.numKeyValueHeads());
        assertEquals(151936, config.vocabSize());
        assertEquals(32768, config.maxPositionEmbeddings());
        assertEquals(4864, config.intermediateSize());
        assertEquals(1e-6f, config.rmsNormEps(), 1e-12f);
        assertEquals(1_000_000.0f, config.ropeTheta(), 1e-3f);
    }

    @Test
    void headDimDerived(@TempDir Path tmp) throws Exception {
        Path configJson = tmp.resolve("config.json");
        Files.writeString(configJson, CONFIG_JSON_0_5B);
        Qwen2Config config = Qwen2Config.load(configJson);

        // headDim = hidden_size / num_attention_heads = 896 / 14 = 64
        assertEquals(64, config.headDim());
    }

    @Test
    void qSizeAndKvSizeDerived(@TempDir Path tmp) throws Exception {
        Path configJson = tmp.resolve("config.json");
        Files.writeString(configJson, CONFIG_JSON_0_5B);
        Qwen2Config config = Qwen2Config.load(configJson);

        // qSize = num_heads * head_dim = 14 * 64 = 896
        assertEquals(896, config.qSize());
        // kvSize = num_kv_heads * head_dim = 2 * 64 = 128
        assertEquals(128, config.kvSize());
    }

    @Test
    void scalesToLargerModel(@TempDir Path tmp) throws Exception {
        Path configJson = tmp.resolve("config.json");
        Files.writeString(configJson, CONFIG_JSON_1_5B);
        Qwen2Config config = Qwen2Config.load(configJson);

        assertEquals(1536, config.hiddenSize());
        assertEquals(12, config.numAttentionHeads());
        assertEquals(28, config.numHiddenLayers());
        assertEquals(2, config.numKeyValueHeads());
        // headDim = 1536 / 12 = 128
        assertEquals(128, config.headDim());
        // qSize = 12 * 128 = 1536
        assertEquals(1536, config.qSize());
        // kvSize = 2 * 128 = 256
        assertEquals(256, config.kvSize());
    }

    @Test
    void ignoresUnknownFields(@TempDir Path tmp) throws Exception {
        String json = """
                {
                  "hidden_size": 896,
                  "num_attention_heads": 14,
                  "num_hidden_layers": 24,
                  "num_key_value_heads": 2,
                  "vocab_size": 151936,
                  "max_position_embeddings": 32768,
                  "intermediate_size": 4864,
                  "rms_norm_eps": 1e-6,
                  "rope_theta": 1000000.0,
                  "some_future_field": true,
                  "another_unknown": [1, 2, 3]
                }
                """;
        Path configJson = tmp.resolve("config.json");
        Files.writeString(configJson, json);

        // Should not throw
        Qwen2Config config = Qwen2Config.load(configJson);
        assertEquals(896, config.hiddenSize());
    }

    @Test
    void missingFileThrowsIoException() {
        assertThrows(java.io.IOException.class, () -> {
            Qwen2Config.load(Path.of("nonexistent/config.json"));
        });
    }

    @Test
    void nullPathThrowsIoException() {
        assertThrows(java.io.IOException.class, () -> {
            Qwen2Config.load(null);
        });
    }
}
