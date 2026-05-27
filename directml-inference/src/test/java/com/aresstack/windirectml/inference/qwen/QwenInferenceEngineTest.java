package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.InferenceException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link QwenInferenceEngine} error handling.
 *
 * <p>Validates error messages for missing/invalid model files without
 * requiring real model weights.
 */
class QwenInferenceEngineTest {

    @Test
    void initializeThrowsWhenModelDirMissing(@TempDir Path tmp) {
        Path nonexistent = tmp.resolve("does-not-exist");
        QwenInferenceEngine engine = new QwenInferenceEngine(nonexistent, 32);

        InferenceException ex = assertThrows(InferenceException.class, engine::initialize);
        assertTrue(ex.getMessage().contains("Cannot initialize Qwen engine"),
                "Error should name the missing file: " + ex.getMessage());
        assertTrue(ex.getMessage().contains(nonexistent.toString()),
                "Error should include missing directory path: " + ex.getMessage());
    }

    @Test
    void initializeThrowsWhenConfigMissing(@TempDir Path tmp) {
        // Empty directory — missing all required files
        QwenInferenceEngine engine = new QwenInferenceEngine(tmp, 32);

        InferenceException ex = assertThrows(InferenceException.class, engine::initialize);
        assertTrue(ex.getMessage().contains("Cannot initialize Qwen engine"),
                "Error should indicate missing files: " + ex.getMessage());
        assertTrue(ex.getMessage().contains("config.json"),
                "Error should include first missing required file name: " + ex.getMessage());
        assertTrue(ex.getMessage().contains(tmp.toString()),
                "Error should include model directory path: " + ex.getMessage());
    }

    @Test
    void isReadyReturnsFalseBeforeInitialize(@TempDir Path tmp) {
        QwenInferenceEngine engine = new QwenInferenceEngine(tmp, 32);
        assertFalse(engine.isReady());
    }

    @Test
    void generateThrowsWhenNotInitialized(@TempDir Path tmp) {
        QwenInferenceEngine engine = new QwenInferenceEngine(tmp, 32);

        var request = com.aresstack.windirectml.inference.InferenceRequest.builder()
                .userPrompt("test")
                .build();

        InferenceException ex = assertThrows(InferenceException.class, () -> engine.generate(request));
        assertTrue(ex.getMessage().contains("not initialized"));
    }

    @Test
    void shutdownIsIdempotent(@TempDir Path tmp) {
        QwenInferenceEngine engine = new QwenInferenceEngine(tmp, 32);
        // Should not throw
        engine.shutdown();
        engine.shutdown();
        assertFalse(engine.isReady());
    }

    @Test
    void initializeFailsEarlyWithUnsupportedFormatMessage(@TempDir Path tmp) throws Exception {
        Files.writeString(tmp.resolve("config.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer.json"), "{\"model\":{\"type\":\"BPE\"}}");
        Files.writeString(tmp.resolve("tokenizer_config.json"), "{}");
        Files.writeString(tmp.resolve("special_tokens_map.json"), "{}");
        Files.writeString(tmp.resolve("model.onnx"), "");
        Files.writeString(tmp.resolve("model.onnx_data"), "");

        QwenInferenceEngine engine = new QwenInferenceEngine(tmp, 32);
        InferenceException ex = assertThrows(InferenceException.class, engine::initialize);
        assertTrue(ex.getMessage().contains("Cannot initialize Qwen engine"), ex.getMessage());
        assertTrue(ex.getMessage().contains("Unsupported Qwen ONNX format"), ex.getMessage());
    }
}
