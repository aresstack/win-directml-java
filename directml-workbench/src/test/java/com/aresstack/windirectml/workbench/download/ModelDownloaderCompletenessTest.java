package com.aresstack.windirectml.workbench.download;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for {@link ModelDownloader#missingRequiredFiles} - the post-download
 * completeness check that detects interrupted / partial installs (ER-2).
 */
class ModelDownloaderCompletenessTest {

    private static final ModelDownloadManifest E5_SMALL_V2 =
            ModelDownloadUrls.manifestForEmbedding("intfloat/e5-small-v2", "e5-small-v2");

    @Test
    void completeDirectoryHasNoMissingRequiredFiles(@TempDir Path tmp) throws IOException {
        writeNonEmpty(tmp, "model.safetensors");
        writeNonEmpty(tmp, "tokenizer.json");
        writeNonEmpty(tmp, "config.json");
        assertEquals(List.of(), ModelDownloader.missingRequiredFiles(E5_SMALL_V2, tmp));
    }

    @Test
    void partialDirectoryReportsMissingConfigJson(@TempDir Path tmp) throws IOException {
        writeNonEmpty(tmp, "model.safetensors");
        writeNonEmpty(tmp, "tokenizer.json");
        // config.json absent -> partial install.
        List<String> missing = ModelDownloader.missingRequiredFiles(E5_SMALL_V2, tmp);
        assertTrue(missing.contains("config.json"), "expected config.json reported missing, got " + missing);
    }

    @Test
    void zeroByteRequiredFileIsReportedMissing(@TempDir Path tmp) throws IOException {
        writeNonEmpty(tmp, "model.safetensors");
        writeNonEmpty(tmp, "tokenizer.json");
        Files.createFile(tmp.resolve("config.json")); // truncated download
        List<String> missing = ModelDownloader.missingRequiredFiles(E5_SMALL_V2, tmp);
        assertTrue(missing.contains("config.json"), "zero-byte config.json must count as missing, got " + missing);
    }

    @Test
    void optionalFilesAreNotRequired(@TempDir Path tmp) throws IOException {
        // Only the required trio present; optional files (vocab.txt, ...) absent.
        writeNonEmpty(tmp, "model.safetensors");
        writeNonEmpty(tmp, "tokenizer.json");
        writeNonEmpty(tmp, "config.json");
        assertEquals(List.of(), ModelDownloader.missingRequiredFiles(E5_SMALL_V2, tmp));
    }

    private static void writeNonEmpty(Path dir, String name) throws IOException {
        Files.writeString(dir.resolve(name), "x");
    }
}
