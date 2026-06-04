package com.aresstack.windirectml.inference.t5;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class T5ConfigTest {

    @TempDir
    Path tempDir;

    @Test
    void loadsCodeT5SmallArchitectureValues() throws Exception {
        Path configJson = tempDir.resolve("config.json");
        Files.writeString(configJson, "{\n" +
                "  \"architectures\": [\"T5ForConditionalGeneration\"],\n" +
                "  \"model_type\": \"t5\",\n" +
                "  \"is_encoder_decoder\": true,\n" +
                "  \"d_model\": 512,\n" +
                "  \"d_kv\": 64,\n" +
                "  \"d_ff\": 2048,\n" +
                "  \"num_layers\": 6,\n" +
                "  \"num_decoder_layers\": 6,\n" +
                "  \"num_heads\": 8,\n" +
                "  \"vocab_size\": 32100,\n" +
                "  \"relative_attention_num_buckets\": 32,\n" +
                "  \"relative_attention_max_distance\": 128,\n" +
                "  \"layer_norm_epsilon\": 1e-6,\n" +
                "  \"decoder_start_token_id\": 0,\n" +
                "  \"eos_token_id\": 2,\n" +
                "  \"pad_token_id\": 0,\n" +
                "  \"feed_forward_proj\": \"relu\"\n" +
                "}\n", StandardCharsets.UTF_8);

        T5Config config = T5Config.load(configJson);

        assertEquals(512, config.modelSize());
        assertEquals(6, config.encoderLayers());
        assertEquals(6, config.effectiveDecoderLayers());
        assertEquals(512, config.attentionInnerSize());
        assertTrue(config.usesTiedWordEmbeddings());
        assertFalse(config.usesGatedFeedForward());
    }

    @Test
    void rejectsDecoderOnlyConfig() throws Exception {
        Path configJson = tempDir.resolve("config.json");
        Files.writeString(configJson, "{\n" +
                "  \"model_type\": \"t5\",\n" +
                "  \"is_encoder_decoder\": false,\n" +
                "  \"d_model\": 4,\n" +
                "  \"d_kv\": 2,\n" +
                "  \"d_ff\": 8,\n" +
                "  \"num_layers\": 1,\n" +
                "  \"num_heads\": 2,\n" +
                "  \"vocab_size\": 6\n" +
                "}\n", StandardCharsets.UTF_8);

        IOExceptionAssertion.assertThrowsWithMessage("encoder-decoder", () -> T5Config.load(configJson));
    }

    private static final class IOExceptionAssertion {
        private IOExceptionAssertion() {
        }

        static void assertThrowsWithMessage(String expected, ThrowingRunnable runnable) {
            Exception error = assertThrows(Exception.class, runnable::run);
            assertTrue(error.getMessage().contains(expected), error.getMessage());
        }
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
