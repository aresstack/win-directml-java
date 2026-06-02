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
        // q4f16 model file via onnx subdir, stored locally as model.onnx by the manifest.
        assertTrue(urls.stream().anyMatch(u -> u.contains("/onnx/model_q4f16.onnx")));
        assertFalse(urls.stream().anyMatch(u -> u.contains("/onnx/model.onnx_data")));
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

    /**
     * Regression: copied Qwen URLs must exactly match the remote paths used by the downloader.
     * If someone updates the config remote paths but forgets URL generation, this test fails.
     */
    @Test
    void qwenCopiedUrlsMatchDownloaderManifestUrls() {
        var config = QwenModelDownloadConfig.DEFAULT;
        var copiedUrls = ModelDownloadUrls.forQwen(config);
        var manifest = ModelDownloadUrls.manifestForQwen(config);
        var manifestUrls = manifest.files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();
        assertEquals(manifestUrls, copiedUrls,
                "Copied URLs must be identical to downloader manifest URLs");
    }

    /**
     * Regression: the q4f16 artifact must not add a dense external-data URL.
     */
    @Test
    void qwenQ4F16UrlUsesSingleFileArtifact() {
        var config = QwenModelDownloadConfig.DEFAULT;
        var urls = ModelDownloadUrls.forQwen(config);
        String expectedModelUrl = "https://huggingface.co/"
                + config.repo() + "/resolve/main/"
                + config.onnxSubdir() + "/model_q4f16.onnx";
        assertTrue(urls.contains(expectedModelUrl),
                "Expected URL " + expectedModelUrl + " not found in: " + urls);
        assertFalse(urls.stream().anyMatch(u -> u.contains("model.onnx_data")),
                "q4f16 default must not require model.onnx_data");
    }

    /**
     * Regression: optional files missing from the list must not corrupt required URLs.
     */
    @Test
    void qwenOptionalFileAbsenceDoesNotAffectRequiredUrls() {
        var configNoOptional = new QwenModelDownloadConfig(
                "onnx-community/Qwen2.5-Coder-0.5B-Instruct",
                "onnx",
                "model_q4f16.onnx",
                "",
                "model.onnx",
                "",
                java.util.List.of("tokenizer.json", "config.json", "tokenizer_config.json", "special_tokens_map.json"),
                java.util.List.of(),  // no optional files
                "qwen2.5-coder-0.5b-directml-int4"
        );
        var urls = ModelDownloadUrls.forQwen(configNoOptional);
        // All required files are still present
        assertTrue(urls.stream().anyMatch(u -> u.endsWith("/onnx/model_q4f16.onnx")));
        assertFalse(urls.stream().anyMatch(u -> u.endsWith("/onnx/model.onnx_data")));
        assertTrue(urls.stream().anyMatch(u -> u.endsWith("/tokenizer.json")));
        assertTrue(urls.stream().anyMatch(u -> u.endsWith("/config.json")));
        assertTrue(urls.stream().anyMatch(u -> u.endsWith("/tokenizer_config.json")));
        assertTrue(urls.stream().anyMatch(u -> u.endsWith("/special_tokens_map.json")));
        // Optional file is absent
        assertFalse(urls.stream().anyMatch(u -> u.endsWith("/added_tokens.json")));
    }
}
