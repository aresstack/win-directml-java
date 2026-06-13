package com.aresstack.windirectml.workbench.download;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Qwen 0.5B download wiring: the runtime download uses the q4f16/ONNX source in the directml-int4
 * directory (not the BF16 SafeTensors path), and special_tokens_map.json is optional so a 404 cannot
 * abort the download.
 */
class QwenDownloadConfigTest {

    @Test
    void quantizedConfigTargetsDirectmlInt4DirWithOnnxSource() {
        QwenModelDownloadConfig config = QwenModelDownloadConfig.DEFAULT_QUANTIZED;
        assertEquals("qwen2.5-coder-0.5b-directml-int4", config.localDirName());
        assertEquals("model_q4f16.onnx", config.modelFile());
        assertEquals(QwenModelDownloadConfig.LOCAL_DIR_NAME, config.localDirName());
    }

    @Test
    void specialTokensMapIsOptionalNotRequired() {
        QwenModelDownloadConfig config = QwenModelDownloadConfig.forVariant(QwenOnnxModelVariant.Q4F16);
        List<String> required = config.rootFiles();
        assertTrue(required.contains("tokenizer.json"));
        assertTrue(required.contains("config.json"));
        assertTrue(required.contains("tokenizer_config.json"));
        assertFalse(required.contains("special_tokens_map.json"),
                "special_tokens_map.json must not be a required root file");
        assertTrue(config.optionalFiles().contains("special_tokens_map.json"),
                "special_tokens_map.json must be optional");
    }

    @Test
    void manifestForQwenMarksSpecialTokensOptionalAndTargetsInt4Dir() {
        ModelDownloadManifest manifest =
                ModelDownloadUrls.manifestForQwen(QwenModelDownloadConfig.DEFAULT_QUANTIZED);
        assertEquals(QwenModelDownloadConfig.LOCAL_DIR_NAME, manifest.localDirName());

        ModelFileDescriptor onnx = find(manifest, "model_q4f16.onnx");
        assertTrue(onnx.required(), "the q4f16 ONNX model file must be a required download");

        ModelFileDescriptor specialTokens = find(manifest, "special_tokens_map.json");
        assertFalse(specialTokens.required(), "special_tokens_map.json must be optional (404 -> skip)");
    }

    private static ModelFileDescriptor find(ModelDownloadManifest manifest, String localFilename) {
        return manifest.files().stream()
                .filter(f -> f.localFilename().equals(localFilename))
                .findFirst()
                .orElseThrow(() -> new AssertionError("no descriptor for " + localFilename
                        + " in " + manifest.files().stream().map(ModelFileDescriptor::localFilename).toList()));
    }
}
