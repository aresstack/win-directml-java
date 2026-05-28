package com.aresstack.windirectml.inference.qwen;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link QwenModelDirValidator}.
 *
 * <p>These tests verify the missing-file diagnostics for Qwen2.5-Coder
 * model directories. No real model weights are needed — the tests use
 * empty placeholder files to check that the validator correctly identifies
 * which file is missing.</p>
 *
 * <p>Modelled after the Phi-3 diagnostics in
 * {@code Phi3InferenceEngineModelDirTest}.</p>
 */
class QwenModelDirValidatorTest {

    @Test
    void nullDirectoryYieldsClearMessage() {
        String msg = QwenModelDirValidator.describeMissingModelFile(null);
        assertNotNull(msg);
        assertTrue(msg.toLowerCase().contains("qwen"), msg);
        assertTrue(msg.contains("null"), msg);
    }

    @Test
    void missingDirectoryYieldsClearMessage(@TempDir Path tmp) {
        Path doesNotExist = tmp.resolve("qwen-not-here");
        String msg = QwenModelDirValidator.describeMissingModelFile(doesNotExist);
        assertNotNull(msg);
        assertTrue(msg.contains("does not exist"), msg);
        assertTrue(msg.contains("qwen-not-here"), msg);
        assertFalse(QwenModelDirValidator.isValidModelDir(doesNotExist));
    }

    @Test
    void emptyDirectoryReportsConfigJsonMissing(@TempDir Path tmp) {
        String msg = QwenModelDirValidator.describeMissingModelFile(tmp);
        assertNotNull(msg);
        assertTrue(msg.contains("config.json"), msg);
    }

    @Test
    void missingTokenizerJsonIsNamed(@TempDir Path tmp) throws Exception {
        Files.writeString(tmp.resolve("config.json"), "{}");
        String msg = QwenModelDirValidator.describeMissingModelFile(tmp);
        assertNotNull(msg);
        assertTrue(msg.contains("tokenizer.json"), msg);
    }

    @Test
    void missingTokenizerConfigJsonIsNamed(@TempDir Path tmp) throws Exception {
        Files.writeString(tmp.resolve("config.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer.json"), "{}");
        String msg = QwenModelDirValidator.describeMissingModelFile(tmp);
        assertNotNull(msg);
        assertTrue(msg.contains("tokenizer_config.json"), msg);
    }

    @Test
    void missingSpecialTokensMapIsNamed(@TempDir Path tmp) throws Exception {
        Files.writeString(tmp.resolve("config.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer_config.json"), "{}");
        String msg = QwenModelDirValidator.describeMissingModelFile(tmp);
        assertNotNull(msg);
        assertTrue(msg.contains("special_tokens_map.json"), msg);
    }

    @Test
    void missingModelOnnxIsNamed(@TempDir Path tmp) throws Exception {
        Files.writeString(tmp.resolve("config.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer_config.json"), "{}");
        Files.writeString(tmp.resolve("special_tokens_map.json"), "{}");
        String msg = QwenModelDirValidator.describeMissingModelFile(tmp);
        assertNotNull(msg);
        assertTrue(msg.contains("model.onnx"), msg);
    }

    @Test
    void missingModelOnnxDataIsNamed(@TempDir Path tmp) throws Exception {
        Files.writeString(tmp.resolve("config.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer_config.json"), "{}");
        Files.writeString(tmp.resolve("special_tokens_map.json"), "{}");
        Files.writeString(tmp.resolve("model.onnx"), "");
        String msg = QwenModelDirValidator.describeMissingModelFile(tmp);
        assertNotNull(msg);
        assertTrue(msg.contains("model.onnx.data"), msg);
    }

    @Test
    void completeDirectoryWithUnparseableOnnxReportsUnsupportedFormat(@TempDir Path tmp) throws Exception {
        Files.writeString(tmp.resolve("config.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer_config.json"), "{}");
        Files.writeString(tmp.resolve("special_tokens_map.json"), "{}");
        Files.writeString(tmp.resolve("model.onnx"), "");
        Files.writeString(tmp.resolve("model.onnx_data"), "");
        String msg = QwenModelDirValidator.describeMissingModelFile(tmp);
        assertNotNull(msg);
        assertTrue(msg.contains("Unsupported Qwen ONNX format"), msg);
        assertFalse(QwenModelDirValidator.isValidModelDir(tmp));
    }

    @Test
    void completeDirectoryWithAltNameAndUnparseableOnnxReportsUnsupportedFormat(@TempDir Path tmp) throws Exception {
        Files.writeString(tmp.resolve("config.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer.json"), "{}");
        Files.writeString(tmp.resolve("tokenizer_config.json"), "{}");
        Files.writeString(tmp.resolve("special_tokens_map.json"), "{}");
        Files.writeString(tmp.resolve("model.onnx"), "");
        Files.writeString(tmp.resolve("model.onnx.data"), "");
        String msg = QwenModelDirValidator.describeMissingModelFile(tmp);
        assertNotNull(msg);
        assertTrue(msg.contains("Unsupported Qwen ONNX format"), msg);
        assertFalse(QwenModelDirValidator.isValidModelDir(tmp));
    }

    @Test
    void resolveExternalDataPathPrefersPrimary(@TempDir Path tmp) throws Exception {
        Files.writeString(tmp.resolve("model.onnx.data"), "primary");
        Path resolved = QwenModelDirValidator.resolveExternalDataPath(tmp);
        assertEquals(tmp.resolve("model.onnx.data"), resolved);
    }

    @Test
    void resolveExternalDataPathFallsBackToAlt(@TempDir Path tmp) throws Exception {
        Files.writeString(tmp.resolve("model.onnx_data"), "alt");
        Path resolved = QwenModelDirValidator.resolveExternalDataPath(tmp);
        assertEquals(tmp.resolve("model.onnx_data"), resolved);
    }

    @Test
    void diagnosticIncludesDirectoryPath(@TempDir Path tmp) {
        String msg = QwenModelDirValidator.describeMissingModelFile(tmp);
        assertNotNull(msg);
        assertTrue(msg.contains(tmp.toString()), msg);
    }
}
