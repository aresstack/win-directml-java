package com.aresstack.windirectml.workbench.download;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies that the Qwen download constants and layout are configured for the
 * onnx-community q4f16 single-file ONNX artifact.
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
        assertFalse(files.contains("model.onnx_data"));
        assertTrue(files.contains("tokenizer.json"));
        assertTrue(files.contains("config.json"));
        assertTrue(files.contains("tokenizer_config.json"));
        assertTrue(files.contains("special_tokens_map.json"));
    }

    @Test
    void defaultConfigUsesOnnxCommunityQ4F16Artifact() {
        var config = QwenModelDownloadConfig.DEFAULT;
        assertEquals("onnx-community/Qwen2.5-Coder-0.5B-Instruct", config.repo());
        assertEquals("onnx", config.onnxSubdir());
        assertEquals("model_q4f16.onnx", config.modelFile());
        assertEquals("", config.externalDataFile());
        assertEquals("model.onnx", config.localModelFile());
        assertEquals("", config.localDataFile());
        assertFalse(config.hasExternalDataFile());
    }

    @Test
    void remotePathsAreCorrect() {
        var config = QwenModelDownloadConfig.DEFAULT;
        assertEquals("onnx/model_q4f16.onnx", config.remoteModelPath());
        assertEquals("", config.remoteDataPath());
    }

    @Test
    void requiredLocalFilesMatchesExpected() {
        var config = QwenModelDownloadConfig.DEFAULT;
        var required = config.requiredLocalFiles();
        assertTrue(required.contains("model.onnx"));
        assertFalse(required.contains("model.onnx_data"));
        assertTrue(required.contains("tokenizer.json"));
        assertTrue(required.contains("config.json"));
        assertTrue(required.contains("tokenizer_config.json"));
        assertTrue(required.contains("special_tokens_map.json"));
        assertFalse(required.contains("added_tokens.json"));
    }

    @Test
    void sidecarConfigStillSupportsExternalData() {
        var config = new QwenModelDownloadConfig(
                "repo/model",
                "onnx",
                "model.onnx",
                "model.onnx_data",
                "model.onnx",
                "model.onnx_data",
                java.util.List.of("tokenizer.json"),
                java.util.List.of(),
                "qwen-test"
        );

        assertTrue(config.hasExternalDataFile());
        assertEquals("onnx/model.onnx_data", config.remoteDataPath());
        assertTrue(config.requiredLocalFiles().contains("model.onnx_data"));
    }
}
