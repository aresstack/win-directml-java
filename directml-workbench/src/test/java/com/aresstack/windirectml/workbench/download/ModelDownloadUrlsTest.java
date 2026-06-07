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


    @Test
    void qwenSafeTensorsManifestUsesCanonicalRepoRootFiles() {
        var manifest = ModelDownloadUrls.manifestForQwenSafeTensors();
        var urls = manifest.files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();

        assertEquals(ModelDownloadUrls.QWEN_SAFETENSORS_LOCAL_DIR, manifest.localDirName());
        assertTrue(urls.contains("https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/resolve/main/model.safetensors"));
        assertTrue(urls.contains("https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/resolve/main/config.json"));
        assertTrue(urls.contains("https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/resolve/main/tokenizer.json"));
        assertEquals("https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/resolve/main/model.safetensors",
                ModelDownloadUrls.selectedQwenSafeTensorsModelUrl());
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
     * Regression: the external data file URL must use the expected underscore filename.
     */
    @Test
    void qwenExternalDataUrlUsesRemoteUnderscoreName() {
        var config = QwenModelDownloadConfig.DEFAULT;
        var urls = ModelDownloadUrls.forQwen(config);
        String expectedDataUrl = "https://huggingface.co/"
                + config.repo() + "/resolve/main/"
                + config.onnxSubdir() + "/" + config.externalDataFile();
        assertTrue(urls.contains(expectedDataUrl),
                "Expected URL " + expectedDataUrl + " not found in: " + urls);
        assertTrue(urls.stream().anyMatch(u -> u.contains("model.onnx_data")),
                "URLs must contain remote model.onnx_data path");
    }

    /**
     * Regression: optional files missing from the list must not corrupt required URLs.
     */
    @Test
    void qwenOptionalFileAbsenceDoesNotAffectRequiredUrls() {
        var configNoOptional = new QwenModelDownloadConfig(
                "onnx-community/Qwen2.5-Coder-0.5B-Instruct",
                "onnx",
                "model.onnx",
                "model.onnx_data",
                "model.onnx",
                "model.onnx_data",
                java.util.List.of("tokenizer.json", "config.json", "tokenizer_config.json", "special_tokens_map.json"),
                java.util.List.of(),  // no optional files
                "qwen2.5-coder-0.5b-directml-int4"
        );
        var urls = ModelDownloadUrls.forQwen(configNoOptional);
        // All required files are still present
        assertTrue(urls.stream().anyMatch(u -> u.endsWith("/onnx/model.onnx")));
        assertTrue(urls.stream().anyMatch(u -> u.endsWith("/onnx/model.onnx_data")));
        assertTrue(urls.stream().anyMatch(u -> u.endsWith("/tokenizer.json")));
        assertTrue(urls.stream().anyMatch(u -> u.endsWith("/config.json")));
        assertTrue(urls.stream().anyMatch(u -> u.endsWith("/tokenizer_config.json")));
        assertTrue(urls.stream().anyMatch(u -> u.endsWith("/special_tokens_map.json")));
        // Optional file is absent
        assertFalse(urls.stream().anyMatch(u -> u.endsWith("/added_tokens.json")));
    }


    @Test
    void qwenCoderScaleUpManifestsContainSafeTensorsAndTokenizerFiles() {
        var manifest15 = ModelDownloadUrls.manifestForQwenCoder1_5BSafeTensors();
        var urls15 = manifest15.files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();
        var manifest3 = ModelDownloadUrls.manifestForQwenCoder3BSafeTensors();
        var urls3 = manifest3.files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();

        assertEquals(ModelDownloadUrls.QWEN_CODER_1_5B_LOCAL_DIR, manifest15.localDirName());
        assertEquals(ModelDownloadUrls.QWEN_CODER_3B_LOCAL_DIR, manifest3.localDirName());
        assertTrue(urls15.contains("https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/model.safetensors"));
        assertTrue(urls15.contains("https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/config.json"));
        assertTrue(urls15.contains("https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/tokenizer.json"));
        assertTrue(urls15.contains("https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/merges.txt"));
        assertTrue(urls15.contains("https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/vocab.json"));
        assertTrue(urls3.contains("https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/model-00001-of-00002.safetensors"));
        assertTrue(urls3.contains("https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/model-00002-of-00002.safetensors"));
        assertTrue(urls3.contains("https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/model.safetensors.index.json"));
        assertTrue(urls3.contains("https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/tokenizer_config.json"));
    }

    @Test
    void smollm2ManifestsContainSafeTensorsAndTokenizerFiles() {
        var manifest135 = ModelDownloadUrls.manifestForSmolLm2_135M();
        var urls135 = manifest135.files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();
        var manifest360 = ModelDownloadUrls.manifestForSmolLm2_360M();
        var urls360 = manifest360.files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();

        assertEquals(ModelDownloadUrls.SMOLLM2_135M_LOCAL_DIR, manifest135.localDirName());
        assertEquals(ModelDownloadUrls.SMOLLM2_360M_LOCAL_DIR, manifest360.localDirName());
        assertTrue(urls135.contains("https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors"));
        assertTrue(urls135.contains("https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json"));
        assertTrue(urls135.contains("https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/merges.txt"));
        assertTrue(urls135.contains("https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/vocab.json"));
        assertTrue(urls360.contains("https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/model.safetensors"));
        assertTrue(urls360.contains("https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/config.json"));
    }

    @Test
    void codeT5SmallManifestContainsTorchCheckpointAndTokenizerFiles() {
        var manifest = ModelDownloadUrls.manifestForCodeT5Small();
        var urls = manifest.files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();

        assertEquals(ModelDownloadUrls.CODET5_SMALL_LOCAL_DIR, manifest.localDirName());
        assertEquals("Salesforce/codet5-small", manifest.modelId());
        assertTrue(urls.contains("https://huggingface.co/Salesforce/codet5-small/resolve/main/pytorch_model.bin"));
        assertTrue(urls.contains("https://huggingface.co/Salesforce/codet5-small/resolve/main/config.json"));
        assertTrue(urls.contains("https://huggingface.co/Salesforce/codet5-small/resolve/main/vocab.json"));
        assertTrue(urls.contains("https://huggingface.co/Salesforce/codet5-small/resolve/main/merges.txt"));
        assertTrue(urls.contains("https://huggingface.co/Salesforce/codet5-small/resolve/main/tokenizer_config.json"));
        assertTrue(urls.contains("https://huggingface.co/Salesforce/codet5-small/resolve/main/special_tokens_map.json"));
    }

    @Test
    void codeT5BaseMultiSumManifestContainsTorchCheckpointAndTokenizerFiles() {
        var manifest = ModelDownloadUrls.manifestForCodeT5BaseMultiSum();
        var urls = manifest.files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();

        assertEquals(ModelDownloadUrls.CODET5_BASE_MULTI_SUM_LOCAL_DIR, manifest.localDirName());
        assertEquals("Salesforce/codet5-base-multi-sum", manifest.modelId());
        assertTrue(urls.contains("https://huggingface.co/Salesforce/codet5-base-multi-sum/resolve/main/pytorch_model.bin"));
        assertTrue(urls.contains("https://huggingface.co/Salesforce/codet5-base-multi-sum/resolve/main/config.json"));
        assertTrue(urls.contains("https://huggingface.co/Salesforce/codet5-base-multi-sum/resolve/main/vocab.json"));
        assertTrue(urls.contains("https://huggingface.co/Salesforce/codet5-base-multi-sum/resolve/main/merges.txt"));
        assertTrue(urls.contains("https://huggingface.co/Salesforce/codet5-base-multi-sum/resolve/main/tokenizer_config.json"));
        assertTrue(urls.contains("https://huggingface.co/Salesforce/codet5-base-multi-sum/resolve/main/special_tokens_map.json"));
    }

    @Test
    void googleT5SmallManifestContainsSafeTensorsAndTokenizerJson() {
        var manifest = ModelDownloadUrls.manifestForGoogleT5Small();
        var urls = manifest.files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();

        assertEquals(ModelDownloadUrls.GOOGLE_T5_SMALL_LOCAL_DIR, manifest.localDirName());
        assertEquals("google-t5/t5-small", manifest.modelId());
        assertTrue(urls.contains("https://huggingface.co/google-t5/t5-small/resolve/main/model.safetensors"));
        assertTrue(urls.contains("https://huggingface.co/google-t5/t5-small/resolve/main/config.json"));
        assertTrue(urls.contains("https://huggingface.co/google-t5/t5-small/resolve/main/tokenizer.json"));
        assertTrue(urls.contains("https://huggingface.co/google-t5/t5-small/resolve/main/spiece.model"));
        assertFalse(urls.contains("https://huggingface.co/google-t5/t5-small/resolve/main/pytorch_model.bin"));
    }

    @Test
    void googleFlanT5SmallManifestContainsSafeTensorsAndTokenizerJson() {
        var manifest = ModelDownloadUrls.manifestForGoogleFlanT5Small();
        var urls = manifest.files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();

        assertEquals(ModelDownloadUrls.GOOGLE_FLAN_T5_SMALL_LOCAL_DIR, manifest.localDirName());
        assertEquals("google/flan-t5-small", manifest.modelId());
        assertTrue(urls.contains("https://huggingface.co/google/flan-t5-small/resolve/main/model.safetensors"));
        assertTrue(urls.contains("https://huggingface.co/google/flan-t5-small/resolve/main/config.json"));
        assertTrue(urls.contains("https://huggingface.co/google/flan-t5-small/resolve/main/tokenizer.json"));
        assertTrue(urls.contains("https://huggingface.co/google/flan-t5-small/resolve/main/spiece.model"));
        assertTrue(urls.contains("https://huggingface.co/google/flan-t5-small/resolve/main/tokenizer_config.json"));
        assertFalse(urls.contains("https://huggingface.co/google/flan-t5-small/resolve/main/pytorch_model.bin"));
    }

}
