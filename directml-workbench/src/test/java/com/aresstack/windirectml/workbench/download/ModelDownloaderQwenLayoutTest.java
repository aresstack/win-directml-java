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
        assertTrue(files.contains("model.onnx_data"));
        assertTrue(files.contains("tokenizer.json"));
        assertTrue(files.contains("config.json"));
        assertTrue(files.contains("tokenizer_config.json"));
        assertTrue(files.contains("special_tokens_map.json"));
    }

    @Test
    void defaultConfigUsesOnnxCommunityRepo() {
        var config = QwenModelDownloadConfig.DEFAULT;
        assertEquals("onnx-community/Qwen2.5-Coder-0.5B-Instruct", config.repo());
        assertEquals("onnx", config.onnxSubdir());
        assertEquals("model.onnx", config.modelFile());
        assertEquals("model.onnx_data", config.externalDataFile());
        assertEquals("model.onnx_data", config.localDataFile());
    }

    @Test
    void remotePathsAreCorrect() {
        var config = QwenModelDownloadConfig.DEFAULT;
        assertEquals("onnx/model.onnx", config.remoteModelPath());
        assertEquals("onnx/model.onnx_data", config.remoteDataPath());
    }

    @Test
    void requiredLocalFilesMatchesExpected() {
        var config = QwenModelDownloadConfig.DEFAULT;
        var required = config.requiredLocalFiles();
        assertTrue(required.contains("model.onnx"));
        assertTrue(required.contains("model.onnx_data"));
        assertTrue(required.contains("tokenizer.json"));
        assertTrue(required.contains("config.json"));
        assertTrue(required.contains("tokenizer_config.json"));
        assertTrue(required.contains("special_tokens_map.json"));
        // added_tokens.json is optional, should not be in required
        assertFalse(required.contains("added_tokens.json"));
    }

    @Test
    void defaultDownloadUsesPrimaryExternalDataName() {
        var config = QwenModelDownloadConfig.DEFAULT;
        assertEquals("model.onnx_data", config.localDataFile());
        assertTrue(config.requiredLocalFiles().contains(config.localDataFile()));
    }
}
