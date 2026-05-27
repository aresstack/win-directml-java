package com.aresstack.windirectml.workbench.download;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies that the Qwen download constants and layout are configured for the
 * onnx-community file structure (model files under {@code onnx/}, config at root).
 */
class ModelDownloaderQwenLayoutTest {

    @Test
    void qwenOnnxSubdirIsOnnx() {
        assertEquals("onnx", ModelDownloader.QWEN_ONNX_SUBDIR,
                "Qwen ONNX files live under onnx/ in the onnx-community repo");
    }

    @Test
    void qwenRequiredFilesIncludesExpectedEntries() {
        var files = ModelDownloader.QWEN_REQUIRED_FILES;
        assertTrue(files.contains("model.onnx"));
        assertTrue(files.contains("model.onnx.data"));
        assertTrue(files.contains("tokenizer.json"));
        assertTrue(files.contains("config.json"));
        assertTrue(files.contains("tokenizer_config.json"));
        assertTrue(files.contains("special_tokens_map.json"));
    }
}
