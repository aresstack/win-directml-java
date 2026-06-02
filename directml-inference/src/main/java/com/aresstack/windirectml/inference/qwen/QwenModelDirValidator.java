package com.aresstack.windirectml.inference.qwen;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Validates that a directory contains the files required by the Qwen2.5-Coder
 * CPU runtime.
 *
 * <p>The Qwen2.5-Coder model requires the following file layout (as defined
 * by issue #100):</p>
 * <ul>
 *   <li>{@code config.json} — model architecture config</li>
 *   <li>{@code tokenizer.json} — HuggingFace BPE tokenizer</li>
 *   <li>{@code tokenizer_config.json} — tokenizer configuration with ChatML template</li>
 *   <li>{@code special_tokens_map.json} — special token definitions</li>
 *   <li>{@code model.onnx} — ONNX model graph</li>
 *   <li>{@code model.onnx.data} — external weight data</li>
 * </ul>
 *
 * <p>This validator is modelled after
 * {@link com.aresstack.windirectml.inference.Phi3InferenceEngine#describeMissingModelFile(Path)}
 * and provides clear diagnostics so users know which file to download.</p>
 */
public final class QwenModelDirValidator {

    /**
     * Primary external data filename (matches onnx-community export).
     */
    public static final String DATA_FILE_PRIMARY = "model.onnx_data";

    /**
     * Alternative external data filename (dot-separated convention).
     */
    public static final String DATA_FILE_ALT = "model.onnx.data";

    /**
     * Required files in download/setup order (aligned with #100 contract).
     */
    private static final String[] REQUIRED_FILES = {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.onnx",
    };

    private QwenModelDirValidator() {
    }

    /**
     * Check whether a directory contains a valid Qwen2.5-Coder model.
     *
     * @param dir model directory to check
     * @return {@code true} if all required files are present
     */
    public static boolean isValidModelDir(Path dir) {
        return describeMissingModelFile(dir) == null;
    }

    /**
     * Diagnose why a directory is not a valid Qwen model directory.
     * Returns {@code null} when the directory is valid; otherwise a
     * human-readable message naming the first missing piece.
     *
     * @param dir model directory to inspect (may be {@code null})
     * @return diagnostic message or {@code null} if valid
     */
    public static String describeMissingModelFile(Path dir) {
        if (dir == null) {
            return "Qwen model directory is null";
        }
        if (!Files.isDirectory(dir)) {
            return "Qwen model directory does not exist: " + dir;
        }
        for (String name : REQUIRED_FILES) {
            if (!Files.exists(dir.resolve(name))) {
                return "Qwen model directory is missing " + name + " (looked in " + dir + ")";
            }
        }
        // Accept either naming convention for external data
        if (!Files.exists(dir.resolve(DATA_FILE_PRIMARY))
                && !Files.exists(dir.resolve(DATA_FILE_ALT))) {
            return "Qwen model directory is missing " + DATA_FILE_PRIMARY
                    + " (or " + DATA_FILE_ALT + ") (looked in " + dir + ")";
        }
        String unsupportedFormat = Qwen2Weights.describeUnsupportedFormat(dir);
        if (unsupportedFormat != null) {
            return unsupportedFormat;
        }
        return null;
    }

    /**
     * Resolve the external data file path, checking both naming conventions.
     *
     * @param modelDir model directory
     * @return path to the external data file
     * @throws java.io.IOException if neither file exists
     */
    public static Path resolveExternalDataPath(Path modelDir) throws java.io.IOException {
        Path primary = modelDir.resolve(DATA_FILE_PRIMARY);
        if (Files.exists(primary)) return primary;
        Path alt = modelDir.resolve(DATA_FILE_ALT);
        if (Files.exists(alt)) return alt;
        throw new java.io.IOException("Required file missing: " + DATA_FILE_PRIMARY
                + " (or " + DATA_FILE_ALT + ") (looked in " + modelDir + ")");
    }
}
