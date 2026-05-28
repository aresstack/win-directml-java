package com.aresstack.windirectml.workbench.download;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link ModelDownloadUrls} – verifies URL computation for all model types.
 */
class ModelDownloadUrlsTest {

    @Test
    void embeddingModelUrlsContainExpectedFiles() {
        var urls = ModelDownloadUrls.forEmbeddingModel("sentence-transformers/all-MiniLM-L6-v2");
        assertFalse(urls.isEmpty());
        assertTrue(urls.contains(
                "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors"));
        assertTrue(urls.contains(
                "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"));
        assertTrue(urls.contains(
                "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json"));
    }

    @Test
    void embeddingModelUrlsIncludeOptionalFiles() {
        var urls = ModelDownloadUrls.forEmbeddingModel("intfloat/e5-small-v2");
        assertTrue(urls.contains(
                "https://huggingface.co/intfloat/e5-small-v2/resolve/main/vocab.txt"));
        assertTrue(urls.contains(
                "https://huggingface.co/intfloat/e5-small-v2/resolve/main/tokenizer_config.json"));
    }

    @Test
    void phi3UrlsContainOnnxFiles() {
        var urls = ModelDownloadUrls.forPhi3();
        assertFalse(urls.isEmpty());
        assertTrue(urls.stream().anyMatch(u -> u.contains("model.onnx") && !u.contains("model.onnx.data")));
        assertTrue(urls.stream().anyMatch(u -> u.contains("model.onnx.data")));
        assertTrue(urls.stream().anyMatch(u -> u.contains("tokenizer.json")));
        // Should use the PHI3_SUBDIR path
        assertTrue(urls.stream().allMatch(u ->
                u.contains("directml/directml-int4-awq-block-128")));
    }

    @Test
    void qwenUrlsContainConfiguredFiles() {
        var config = QwenModelDownloadConfig.DEFAULT;
        var urls = ModelDownloadUrls.forQwen(config);
        assertFalse(urls.isEmpty());
        // Model and data files via onnx subdir
        assertTrue(urls.stream().anyMatch(u ->
                u.contains("/onnx/model.onnx") && !u.contains("model.onnx_data")));
        assertTrue(urls.stream().anyMatch(u -> u.contains("/onnx/model.onnx_data")));
        // Root files (tokenizer, config)
        assertTrue(urls.stream().anyMatch(u ->
                u.endsWith("/tokenizer.json")));
        assertTrue(urls.stream().anyMatch(u ->
                u.endsWith("/config.json")));
        // Optional files included
        assertTrue(urls.stream().anyMatch(u ->
                u.endsWith("/added_tokens.json")));
    }

    @Test
    void allUrlsStartWithHuggingFaceBase() {
        var embUrls = ModelDownloadUrls.forEmbeddingModel("test/repo");
        for (String url : embUrls) {
            assertTrue(url.startsWith("https://huggingface.co/"), "URL should start with HF base: " + url);
        }
    }
}
