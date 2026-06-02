package com.aresstack.windirectml.inference.qwen;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Validates that a directory contains the files required by the Qwen2.5-Coder
 * runtime.
 *
 * <p>The loader supports both common ONNX layouts:</p>
 * <ul>
 *   <li>single-file ONNX artifacts, for example {@code model_q4f16.onnx} saved as {@code model.onnx}</li>
 *   <li>ONNX graph plus external data sidecar, for example {@code model.onnx_data}</li>
 * </ul>
 */
public final class QwenModelDirValidator {

    /**
     * Primary external data filename (matches onnx-community dense export).
     */
    public static final String DATA_FILE_PRIMARY = "model.onnx_data";

    /**
     * Alternative external data filename (dot-separated convention).
     */
    public static final String DATA_FILE_ALT = "model.onnx.data";

    /**
     * Required files in download/setup order. External data is layout-dependent.
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
     * @return {@code true} if all required files are present and the ONNX layout is supported
     */
    public static boolean isValidModelDir(Path dir) {
        return describeMissingModelFile(dir) == null;
    }

    /**
     * Diagnose why a directory is not a valid Qwen model directory.
     * Returns {@code null} when the directory is valid.
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
        String unsupportedFormat = Qwen2Weights.describeUnsupportedFormat(dir);
        if (unsupportedFormat != null) {
            return unsupportedFormat;
        }
        return null;
    }

    /**
     * Resolve the external data file path when one is present.
     *
     * @param modelDir model directory
     * @return path to the external data file, or {@code null} for single-file ONNX artifacts
     */
    public static Path resolveExternalDataPathIfPresent(Path modelDir) {
        Path primary = modelDir.resolve(DATA_FILE_PRIMARY);
        if (Files.exists(primary)) return primary;
        Path alt = modelDir.resolve(DATA_FILE_ALT);
        if (Files.exists(alt)) return alt;
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
        Path resolved = resolveExternalDataPathIfPresent(modelDir);
        if (resolved != null) return resolved;
        throw new java.io.IOException("Required file missing: " + DATA_FILE_PRIMARY
                + " (or " + DATA_FILE_ALT + ") (looked in " + modelDir + ")");
    }
}
