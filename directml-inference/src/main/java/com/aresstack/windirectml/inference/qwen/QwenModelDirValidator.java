package com.aresstack.windirectml.inference.qwen;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Validates that a directory contains the files required by the Qwen2.5-Coder
 * runtime.
 */
public final class QwenModelDirValidator {

    public static final String DEFAULT_MODEL_FILE = "model.onnx";

    /**
     * Primary external data filename (matches onnx-community export).
     */
    public static final String DATA_FILE_PRIMARY = "model.onnx_data";

    /**
     * Alternative external data filename (dot-separated convention).
     */
    public static final String DATA_FILE_ALT = "model.onnx.data";

    /**
     * Required metadata files in download/setup order.
     */
    private static final String[] REQUIRED_METADATA_FILES = {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json"
    };

    private QwenModelDirValidator() {
    }

    public static boolean isValidModelDir(Path dir) {
        return describeMissingModelFile(dir) == null;
    }

    public static boolean isValidModelDir(Path dir, String modelFileName) {
        return describeMissingModelFile(dir, modelFileName) == null;
    }

    public static String describeMissingModelFile(Path dir) {
        return describeMissingModelFile(dir, DEFAULT_MODEL_FILE);
    }

    public static String describeMissingModelFile(Path dir, String modelFileName) {
        String missing = describeMissingRequiredFiles(dir, modelFileName);
        if (missing != null) {
            return missing;
        }

        String safeModelFile;
        try {
            safeModelFile = normalizeModelFileName(modelFileName);
        } catch (IllegalArgumentException ex) {
            return "Invalid Qwen ONNX filename: " + ex.getMessage();
        }

        String unsupportedFormat = Qwen2Weights.describeUnsupportedFormat(dir, safeModelFile);
        if (unsupportedFormat != null) {
            return unsupportedFormat;
        }
        return null;
    }

    /**
     * Fast validation used on the UI/runtime hot startup path.
     *
     * <p>This intentionally checks only required files and filename shape. Full
     * ONNX layout validation happens during the single model import in
     * {@link Qwen2Weights#load(Path, Qwen2Config, String)}. Keeping the heavy
     * format check out of this method avoids parsing the same 500+ MiB ONNX
     * file multiple times before model load.</p>
     */
    public static String describeMissingRequiredFiles(Path dir, String modelFileName) {
        if (dir == null) {
            return "Qwen model directory is null";
        }
        if (!Files.isDirectory(dir)) {
            return "Qwen model directory does not exist: " + dir;
        }
        for (String name : REQUIRED_METADATA_FILES) {
            if (!Files.exists(dir.resolve(name))) {
                return "Qwen model directory is missing " + name + " (looked in " + dir + ")";
            }
        }

        String safeModelFile;
        try {
            safeModelFile = normalizeModelFileName(modelFileName);
        } catch (IllegalArgumentException ex) {
            return "Invalid Qwen ONNX filename: " + ex.getMessage();
        }

        if (!Files.exists(dir.resolve(safeModelFile))) {
            return "Qwen model directory is missing " + safeModelFile + " (looked in " + dir + ")";
        }
        if (requiresExternalDataFile(safeModelFile)
                && !Files.exists(dir.resolve(DATA_FILE_PRIMARY))
                && !Files.exists(dir.resolve(DATA_FILE_ALT))) {
            return "Qwen model directory is missing " + DATA_FILE_PRIMARY
                    + " (or " + DATA_FILE_ALT + ") (looked in " + dir + ")";
        }
        return null;
    }

    public static Path resolveExternalDataPath(Path modelDir) throws java.io.IOException {
        Path primary = modelDir.resolve(DATA_FILE_PRIMARY);
        if (Files.exists(primary)) {
            return primary;
        }
        Path alt = modelDir.resolve(DATA_FILE_ALT);
        if (Files.exists(alt)) {
            return alt;
        }
        throw new java.io.IOException("Required file missing: " + DATA_FILE_PRIMARY
                + " (or " + DATA_FILE_ALT + ") (looked in " + modelDir + ")");
    }

    public static boolean requiresExternalDataFile(String modelFileName) {
        return DEFAULT_MODEL_FILE.equals(normalizeModelFileName(modelFileName));
    }

    public static String normalizeModelFileName(String modelFileName) {
        String trimmed = modelFileName == null || modelFileName.trim().isEmpty()
                ? DEFAULT_MODEL_FILE
                : modelFileName.trim();
        Path path = Path.of(trimmed);
        if (path.getNameCount() != 1 || trimmed.contains("/") || trimmed.contains("\\")) {
            throw new IllegalArgumentException(trimmed + " is not a plain file name");
        }
        if (!trimmed.endsWith(".onnx")) {
            throw new IllegalArgumentException(trimmed + " does not end with .onnx");
        }
        return trimmed;
    }
}
