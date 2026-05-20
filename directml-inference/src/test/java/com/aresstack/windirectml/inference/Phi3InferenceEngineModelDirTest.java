package com.aresstack.windirectml.inference;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit-test for {@link Phi3InferenceEngine#describeMissingModelFile(Path)}.
 *
 * <p>The Phi-3 summarizer is an optional, experimental sidecar feature.
 * When the model directory is incomplete the sidecar must report
 * <em>which file is missing</em> so the user knows what to download.</p>
 */
class Phi3InferenceEngineModelDirTest {

    @Test
    void nullDirectoryYieldsClearMessage() {
        String msg = Phi3InferenceEngine.describeMissingModelFile(null);
        assertNotNull(msg);
        assertTrue(msg.toLowerCase().contains("phi-3"));
    }

    @Test
    void missingDirectoryYieldsClearMessage(@TempDir Path tmp) {
        Path doesNotExist = tmp.resolve("definitely-not-here");
        String msg = Phi3InferenceEngine.describeMissingModelFile(doesNotExist);
        assertNotNull(msg);
        assertTrue(msg.contains("does not exist"), msg);
        assertTrue(msg.contains("definitely-not-here"), msg);
    }

    @Test
    void emptyDirectoryReportsConfigJsonMissing(@TempDir Path tmp) {
        String msg = Phi3InferenceEngine.describeMissingModelFile(tmp);
        assertNotNull(msg);
        assertTrue(msg.contains("config.json"), msg);
    }

    @Test
    void missingTokenizerJsonIsNamed(@TempDir Path tmp) throws Exception {
        Files.writeString(tmp.resolve("config.json"), "{}");
        String msg = Phi3InferenceEngine.describeMissingModelFile(tmp);
        assertNotNull(msg);
        assertTrue(msg.contains("tokenizer.json"), msg);
    }

    @Test
    void missingModelOnnxIsNamed(@TempDir Path tmp) throws Exception {
        Files.writeString(tmp.resolve("config.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer.json"), "{}");
        String msg = Phi3InferenceEngine.describeMissingModelFile(tmp);
        assertNotNull(msg);
        assertTrue(msg.contains("model.onnx"), msg);
    }

    @Test
    void missingModelOnnxDataIsNamed(@TempDir Path tmp) throws Exception {
        Files.writeString(tmp.resolve("config.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer.json"), "{}");
        Files.writeString(tmp.resolve("model.onnx"), "");
        String msg = Phi3InferenceEngine.describeMissingModelFile(tmp);
        assertNotNull(msg);
        assertTrue(msg.contains("model.onnx.data"), msg);
    }

    @Test
    void completeDirectoryReturnsNull(@TempDir Path tmp) throws Exception {
        Files.writeString(tmp.resolve("config.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer.json"), "{}");
        Files.writeString(tmp.resolve("model.onnx"), "");
        Files.writeString(tmp.resolve("model.onnx.data"), "");
        assertNull(Phi3InferenceEngine.describeMissingModelFile(tmp));
        assertTrue(Phi3InferenceEngine.isValidModelDir(tmp));
    }
}
