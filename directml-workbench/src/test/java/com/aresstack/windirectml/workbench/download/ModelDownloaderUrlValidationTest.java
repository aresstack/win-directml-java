package com.aresstack.windirectml.workbench.download;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ModelDownloaderUrlValidationTest {

    @Test
    void requiredBlankUrlThrowsClearIOException(@TempDir Path tempDir) {
        var manifest = new ModelDownloadManifest(
                "test-model",
                "test-model",
                List.of(new ModelFileDescriptor(
                        "Model",
                        true,
                        "https://example.com/model.onnx",
                        "   ",
                        "model.onnx"
                )));

        IOException ex = assertThrows(IOException.class, () ->
                ModelDownloader.downloadFromManifest(manifest, tempDir, false, s -> {}));
        assertTrue(ex.getMessage().contains("Invalid required download URL for model.onnx"),
                "Expected clear required URL error, got: " + ex.getMessage());
    }

    @Test
    void optionalBlankUrlIsSkipped(@TempDir Path tempDir) throws IOException, InterruptedException {
        var manifest = new ModelDownloadManifest(
                "test-model",
                "test-model",
                List.of(new ModelFileDescriptor(
                        "Optional file",
                        false,
                        "https://example.com/optional.json",
                        "   ",
                        "optional.json"
                )));

        ModelDownloader.downloadFromManifest(manifest, tempDir, false, s -> {});
        assertFalse(Files.exists(tempDir.resolve("optional.json")));
    }

    @Test
    void malformedUrlBecomesIOExceptionWithFilenameAndUrl(@TempDir Path tempDir) {
        var manifest = new ModelDownloadManifest(
                "test-model",
                "test-model",
                List.of(new ModelFileDescriptor(
                        "Model",
                        true,
                        "https://example.com/model.onnx",
                        "ht!tp://bad url",
                        "model.onnx"
                )));

        IOException ex = assertThrows(IOException.class, () ->
                ModelDownloader.downloadFromManifest(manifest, tempDir, false, s -> {}));
        assertTrue(ex.getMessage().contains("model.onnx"), "Expected filename in error");
        assertTrue(ex.getMessage().contains("ht!tp://bad url"), "Expected invalid URL in error");
    }
}
