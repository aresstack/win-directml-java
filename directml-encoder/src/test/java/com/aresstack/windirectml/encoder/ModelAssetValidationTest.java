package com.aresstack.windirectml.encoder;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Device-free tests for {@link ModelAssetValidation}: each asset state must
 * produce a distinct, actionable message.
 */
class ModelAssetValidationTest {

    private static final List<String> REQUIRED = List.of("config.json", "tokenizer.json", "model.safetensors");
    private static final String HINT = "Repair: re-download with Force.";

    @Test
    void missingDirectoryReportsNotDownloaded(@TempDir Path tmp) {
        Path absent = tmp.resolve("never-downloaded");
        EmbeddingException ex = assertThrows(EmbeddingException.class,
                () -> ModelAssetValidation.requireModelFiles(absent, "E5", REQUIRED, HINT));
        assertTrue(ex.getMessage().contains("not been downloaded"),
                "Expected a not-downloaded message, got: " + ex.getMessage());
        assertTrue(ex.getMessage().contains(HINT), "Repair hint must be included");
    }

    @Test
    void presentDirectoryMissingRequiredFileReportsIncomplete(@TempDir Path tmp) throws IOException {
        // tokenizer.json + model.safetensors present, config.json absent -> incomplete.
        Files.writeString(tmp.resolve("tokenizer.json"), "{}");
        Files.writeString(tmp.resolve("model.safetensors"), "x");
        EmbeddingException ex = assertThrows(EmbeddingException.class,
                () -> ModelAssetValidation.requireModelFiles(tmp, "E5", REQUIRED, HINT));
        assertTrue(ex.getMessage().contains("incomplete"),
                "Expected an incomplete message, got: " + ex.getMessage());
        assertTrue(ex.getMessage().contains("config.json"),
                "Message must name the missing file, got: " + ex.getMessage());
    }

    @Test
    void zeroByteRequiredFileReportsCorrupt(@TempDir Path tmp) throws IOException {
        Files.writeString(tmp.resolve("config.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer.json"), "{}");
        Files.createFile(tmp.resolve("model.safetensors")); // zero bytes
        EmbeddingException ex = assertThrows(EmbeddingException.class,
                () -> ModelAssetValidation.requireModelFiles(tmp, "Reranker", REQUIRED, HINT));
        assertTrue(ex.getMessage().contains("corrupt"),
                "Expected a corrupt (zero-byte) message, got: " + ex.getMessage());
        assertTrue(ex.getMessage().contains("model.safetensors"),
                "Message must name the zero-byte file, got: " + ex.getMessage());
    }

    @Test
    void completeDirectoryPasses(@TempDir Path tmp) throws IOException {
        Files.writeString(tmp.resolve("config.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer.json"), "{}");
        Files.writeString(tmp.resolve("model.safetensors"), "weights");
        assertDoesNotThrow(() -> ModelAssetValidation.requireModelFiles(tmp, "E5", REQUIRED, HINT));
    }
}
