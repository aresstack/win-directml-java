package com.aresstack.windirectml.workbench.download;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link ModelDownloadManifest}, {@link ModelFileDescriptor},
 * and {@link DownloadOverrideStore}.
 */
class DownloadManifestTest {

    @Test
    void manifestForEmbeddingContainsExpectedFiles() {
        var manifest = ModelDownloadUrls.manifestForEmbedding(
                "sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2");
        assertEquals("all-MiniLM-L6-v2", manifest.modelId());
        assertEquals("all-MiniLM-L6-v2", manifest.localDirName());
        assertFalse(manifest.files().isEmpty());
        // Required files
        assertTrue(manifest.files().stream().anyMatch(f ->
                f.localFilename().equals("model.safetensors") && f.required()));
        assertTrue(manifest.files().stream().anyMatch(f ->
                f.localFilename().equals("tokenizer.json") && f.required()));
        // Optional files
        assertTrue(manifest.files().stream().anyMatch(f ->
                f.localFilename().equals("vocab.txt") && !f.required()));
    }

    @Test
    void manifestForPhi3ContainsExpectedFiles() {
        var manifest = ModelDownloadUrls.manifestForPhi3();
        assertEquals("phi-3-mini-4k-instruct-onnx", manifest.modelId());
        assertTrue(manifest.files().stream().anyMatch(f ->
                f.localFilename().equals("model.onnx") && f.required()));
    }

    @Test
    void manifestForQwenContainsExpectedFiles() {
        var manifest = ModelDownloadUrls.manifestForQwen(QwenModelDownloadConfig.DEFAULT);
        assertEquals("qwen2.5-coder-0.5b-directml-int4", manifest.modelId());
        assertTrue(manifest.files().stream().anyMatch(f ->
                f.localFilename().equals("model.onnx") && f.required()));
        assertTrue(manifest.files().stream().anyMatch(f ->
                f.localFilename().equals("model.onnx_data") && f.required()));
        assertTrue(manifest.files().stream().anyMatch(f ->
                f.localFilename().equals("model.onnx_data")
                        && f.defaultUrl().endsWith("/onnx/model.onnx_data")
                        && f.required()));
        assertTrue(manifest.files().stream().anyMatch(f ->
                f.localFilename().equals("added_tokens.json") && !f.required()));
    }


    @Test
    void qwenDefaultQuantizedUrlsMatchHistoricalDownloaderDefaults() {
        var manifest = ModelDownloadUrls.manifestForQwen(QwenModelDownloadConfig.DEFAULT_QUANTIZED);
        String base = "https://huggingface.co/onnx-community/Qwen2.5-Coder-0.5B-Instruct/resolve/main/";
        assertEquals("qwen2.5-coder-0.5b-directml-int4", manifest.modelId());
        assertDescriptor(manifest, "model_q4f16.onnx", true, base + "onnx/model_q4f16.onnx");
        assertDescriptor(manifest, "tokenizer.json", true, base + "tokenizer.json");
        assertDescriptor(manifest, "config.json", true, base + "config.json");
        assertDescriptor(manifest, "tokenizer_config.json", true, base + "tokenizer_config.json");
        assertDescriptor(manifest, "special_tokens_map.json", false, base + "special_tokens_map.json");
        assertDescriptor(manifest, "added_tokens.json", false, base + "added_tokens.json");
        assertDescriptor(manifest, "generation_config.json", false, base + "generation_config.json");
    }

    @Test
    void qwenDenseDefaultKeepsExternalDataUrl() {
        var manifest = ModelDownloadUrls.manifestForQwen(QwenModelDownloadConfig.DEFAULT);
        String base = "https://huggingface.co/onnx-community/Qwen2.5-Coder-0.5B-Instruct/resolve/main/onnx/";
        assertDescriptor(manifest, "model.onnx", true, base + "model.onnx");
        assertDescriptor(manifest, "model.onnx_data", true, base + "model.onnx_data");
    }

    @Test
    void qwenSafeTensorsSpecialTokensMapIsOptionalOverrideFile() {
        var manifest = ModelDownloadUrls.manifestForQwenSafeTensors();
        String base = "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/resolve/main/";
        assertDescriptor(manifest, "special_tokens_map.json", false, base + "special_tokens_map.json");
    }

    @Test
    void manifestForGemma3InstructContainsGatedModelFiles() {
        var manifest = ModelDownloadUrls.manifestForGemma3_270MInstruct();
        assertEquals("google/gemma-3-270m-it", manifest.modelId());
        assertEquals("gemma-3-270m-it", manifest.localDirName());
        assertTrue(manifest.files().stream().anyMatch(f ->
                f.localFilename().equals("model.safetensors") && f.required()));
        assertTrue(manifest.files().stream().anyMatch(f ->
                f.localFilename().equals("tokenizer.model") && f.required()));
        assertTrue(manifest.files().stream().anyMatch(f ->
                f.localFilename().equals("chat_template.jinja") && !f.required()));
    }

    @Test
    void qwenOverridesApplyToSelectedModelFileUrl(@TempDir Path tempDir) throws IOException {
        Path storeFile = tempDir.resolve("download-overrides.json");
        var store = new DownloadOverrideStore(storeFile);
        var manifest = ModelDownloadUrls.manifestForQwen(QwenModelDownloadConfig.DEFAULT_QUANTIZED);
        String customUrl = "https://example.invalid/qwen/model_q4f16.onnx";

        store.storeOverrides(manifest.withFileUrl(0, customUrl));

        var reloadedStore = new DownloadOverrideStore(storeFile);
        reloadedStore.load();
        var reloadedManifest = reloadedStore.applyOverrides(manifest);
        assertEquals(customUrl, reloadedManifest.files().get(0).currentUrl());
        assertEquals(manifest.files().get(0).defaultUrl(), reloadedManifest.files().get(0).defaultUrl());
    }

    @Test
    void overrideStorePreservesTokenWhenUrlsAreUpdated(@TempDir Path tempDir) throws IOException {
        Path storeFile = tempDir.resolve("download-overrides.json");
        var store = new DownloadOverrideStore(storeFile);
        var manifest = ModelDownloadUrls.manifestForGemma3_270MInstruct();

        store.storeAccessSettings(manifest.modelId(), new DownloadAccessSettings("hf_secret_token"));
        store.storeOverrides(manifest.withFileUrl(0, "https://example.invalid/model.safetensors"));

        var reloadedStore = new DownloadOverrideStore(storeFile);
        reloadedStore.load();
        var reloadedManifest = reloadedStore.applyOverrides(manifest);
        assertEquals("https://example.invalid/model.safetensors", reloadedManifest.files().get(0).currentUrl());
        assertEquals("hf_secret_token", reloadedStore.accessSettings(manifest.modelId()).huggingFaceToken());
    }

    @Test
    void overrideStoreReadsVersionedDownloadSettings(@TempDir Path tempDir) throws IOException {
        Path storeFile = tempDir.resolve("download-overrides.json");
        String json = """
                {
                  "urlOverrides": {
                    "google/gemma-3-270m-it": {
                      "model.safetensors": "https://example.com/gemma.safetensors"
                    }
                  },
                  "huggingFaceTokens": {
                    "google/gemma-3-270m-it": "hf_secret_token"
                  }
                }
                """;
        Files.writeString(storeFile, json);

        var store = new DownloadOverrideStore(storeFile);
        store.load();
        var manifest = ModelDownloadUrls.manifestForGemma3_270MInstruct();
        var result = store.applyOverrides(manifest);

        assertEquals("https://example.com/gemma.safetensors", result.files().get(0).currentUrl());
        assertEquals("hf_secret_token", store.accessSettings(manifest.modelId()).huggingFaceToken());
    }

    @Test
    void fileDescriptorWithCurrentUrlCreatesNewInstance() {
        var desc = new ModelFileDescriptor("test.json", true,
                "http://example.com/test.json", "http://example.com/test.json", "test.json");
        var updated = desc.withCurrentUrl("http://other.com/test.json");
        assertEquals("http://other.com/test.json", updated.currentUrl());
        assertEquals("http://example.com/test.json", updated.defaultUrl());
        assertEquals("http://example.com/test.json", desc.currentUrl()); // original unchanged
    }

    @Test
    void manifestWithAllUrlsReplacesCorrectly() {
        var manifest = ModelDownloadUrls.manifestForPhi3();
        int size = manifest.files().size();
        var newUrls = new java.util.ArrayList<String>();
        for (int i = 0; i < size; i++) {
            newUrls.add("http://custom/" + i);
        }
        var updated = manifest.withAllUrls(newUrls);
        for (int i = 0; i < size; i++) {
            assertEquals("http://custom/" + i, updated.files().get(i).currentUrl());
            // Default should be unchanged
            assertEquals(manifest.files().get(i).defaultUrl(), updated.files().get(i).defaultUrl());
        }
    }

    @Test
    void overrideStoreRoundTrip(@TempDir Path tempDir) throws IOException {
        Path storeFile = tempDir.resolve("overrides.json");
        var store = new DownloadOverrideStore(storeFile);

        var manifest = ModelDownloadUrls.manifestForPhi3();
        // Override one URL
        var updated = manifest.withFileUrl(0, "http://custom-url/model.onnx");
        store.storeOverrides(updated);

        // Reload from disk
        var store2 = new DownloadOverrideStore(storeFile);
        store2.load();
        var reloaded = store2.applyOverrides(manifest);
        assertEquals("http://custom-url/model.onnx", reloaded.files().get(0).currentUrl());
        // Other files unchanged
        assertEquals(manifest.files().get(1).defaultUrl(), reloaded.files().get(1).currentUrl());
    }

    @Test
    void overrideStoreToleratesMissingFile(@TempDir Path tempDir) {
        Path storeFile = tempDir.resolve("nonexistent.json");
        var store = new DownloadOverrideStore(storeFile);
        store.load(); // should not throw

        var manifest = ModelDownloadUrls.manifestForPhi3();
        var result = store.applyOverrides(manifest);
        // No overrides applied
        assertEquals(manifest.files().get(0).currentUrl(), result.files().get(0).currentUrl());
    }

    @Test
    void overrideStoreToleratesCorruptFile(@TempDir Path tempDir) throws IOException {
        Path storeFile = tempDir.resolve("corrupt.json");
        Files.writeString(storeFile, "not valid json {{{");
        var store = new DownloadOverrideStore(storeFile);
        store.load(); // should not throw, just log warning

        var manifest = ModelDownloadUrls.manifestForPhi3();
        var result = store.applyOverrides(manifest);
        assertEquals(manifest.files().get(0).currentUrl(), result.files().get(0).currentUrl());
    }

    @Test
    void overrideStoreOnlyPersistsDifferences(@TempDir Path tempDir) throws IOException {
        Path storeFile = tempDir.resolve("overrides.json");
        var store = new DownloadOverrideStore(storeFile);

        // Store manifest with no changes (all URLs match defaults)
        var manifest = ModelDownloadUrls.manifestForPhi3();
        store.storeOverrides(manifest);

        // File should be essentially empty or contain empty object
        String content = Files.readString(storeFile);
        assertFalse(content.contains("model.onnx"),
                "Should not persist entries where current==default");
    }

    @Test
    void overrideStoreTreatsBlankOverrideAsNoOverride(@TempDir Path tempDir) throws IOException {
        Path storeFile = tempDir.resolve("overrides.json");
        var store = new DownloadOverrideStore(storeFile);

        var manifest = ModelDownloadUrls.manifestForPhi3().withFileUrl(0, "   ");
        store.storeOverrides(manifest);

        String content = Files.readString(storeFile);
        assertFalse(content.contains("model.onnx"),
                "Blank override should not be persisted");
    }

    @Test
    void overrideStoreParsesJsonEscapesInUrls(@TempDir Path tempDir) throws IOException {
        Path storeFile = tempDir.resolve("overrides.json");
        String json = "{\n"
                + "  \"phi-3-mini-4k-instruct-onnx\": {\n"
                + "    \"model.onnx\": \"https:\\/\\/example.com\\/artifact?x=1\\u0026y=2\"\n"
                + "  }\n"
                + "}\n";
        Files.writeString(storeFile, json);

        var store = new DownloadOverrideStore(storeFile);
        store.load();
        var manifest = ModelDownloadUrls.manifestForPhi3();
        var result = store.applyOverrides(manifest);

        assertEquals("https://example.com/artifact?x=1&y=2",
                result.files().stream()
                        .filter(f -> f.localFilename().equals("model.onnx"))
                        .findFirst()
                        .orElseThrow()
                        .currentUrl());
    }

    @Test
    void defaultModelRootEndsWithExpectedPath() {
        Path root = DownloadOverrideStore.defaultModelRoot();
        assertTrue(root.toString().contains(".directml"),
                "Expected .directml in path: " + root);
        assertTrue(root.endsWith(Path.of(".directml", "model")),
                "Expected path to end with .directml/model: " + root);
    }

    private static String hf(String repo, String file) {
        return "https://huggingface.co/" + repo + "/resolve/main/" + file;
    }

    private static void assertDescriptor(ModelDownloadManifest manifest, String localFilename,
                                         boolean required, String expectedUrl) {
        ModelFileDescriptor descriptor = manifest.files().stream()
                .filter(file -> file.localFilename().equals(localFilename))
                .findFirst()
                .orElseThrow(() -> new AssertionError("Missing descriptor: " + localFilename));
        assertEquals(required, descriptor.required(), localFilename + " required flag");
        assertEquals(expectedUrl, descriptor.defaultUrl(), localFilename + " default URL");
        assertEquals(expectedUrl, descriptor.currentUrl(), localFilename + " current URL");
    }
}
