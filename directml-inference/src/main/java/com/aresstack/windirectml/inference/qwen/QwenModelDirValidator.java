package com.aresstack.windirectml.inference.qwen;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;

/**
 * Validates that a directory contains the files required by the Qwen2.5-Coder
 * runtime.
 *
 * <p>The runtime can use different Hugging Face ONNX filenames from the same
 * directory. The active filename is stored in {@link #SELECTED_ONNX_FILE}.
 * If the selection file is missing, {@code model.onnx} is used for backward
 * compatibility with the original external-data layout.</p>
 */
public final class QwenModelDirValidator {

    /**
     * Default ONNX filename used when no explicit selection exists.
     */
    public static final String DEFAULT_ONNX_FILE = "model.onnx";

    /**
     * Text file containing the active ONNX filename, e.g. {@code model_q4f16.onnx}.
     */
    public static final String SELECTED_ONNX_FILE = "qwen-selected-onnx.txt";

    /**
     * Optional system property for one-off runtime experiments.
     */
    public static final String ONNX_FILE_PROPERTY = "qwen.onnx.file";

    /**
     * Primary external data filename (matches onnx-community export).
     */
    public static final String DATA_FILE_PRIMARY = "model.onnx_data";

    /**
     * Alternative external data filename (dot-separated convention).
     */
    public static final String DATA_FILE_ALT = "model.onnx.data";

    /**
     * Required non-weight support files in download/setup order.
     */
    private static final String[] REQUIRED_SUPPORT_FILES = {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json"
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
        for (String name : REQUIRED_SUPPORT_FILES) {
            if (!Files.exists(dir.resolve(name))) {
                return "Qwen model directory is missing " + name + " (looked in " + dir + ")";
            }
        }

        String onnxFile = selectedOnnxFilename(dir);
        if (!isSafeOnnxFilename(onnxFile)) {
            return "Qwen ONNX selection is invalid: " + onnxFile
                    + " (selection file: " + dir.resolve(SELECTED_ONNX_FILE) + ")";
        }
        if (!Files.exists(dir.resolve(onnxFile))) {
            return "Qwen model directory is missing selected ONNX model " + onnxFile
                    + " (looked in " + dir + ")";
        }

        if (DEFAULT_ONNX_FILE.equals(onnxFile)
                && !Files.exists(dir.resolve(DATA_FILE_PRIMARY))
                && !Files.exists(dir.resolve(DATA_FILE_ALT))) {
            return "Qwen model directory is missing " + DATA_FILE_PRIMARY
                    + " (or " + DATA_FILE_ALT + ") for " + DEFAULT_ONNX_FILE
                    + " (looked in " + dir + ")";
        }

        String unsupportedFormat = Qwen2Weights.describeUnsupportedFormat(dir);
        if (unsupportedFormat != null) {
            return unsupportedFormat;
        }
        return null;
    }

    /**
     * Resolve the active ONNX model path using the selection file.
     *
     * @param modelDir model directory
     * @return selected ONNX file path
     * @throws IOException if the configured filename is unsafe or missing
     */
    public static Path resolveOnnxModelPath(Path modelDir) throws IOException {
        String onnxFile = selectedOnnxFilename(modelDir);
        if (!isSafeOnnxFilename(onnxFile)) {
            throw new IOException("Invalid Qwen ONNX filename: " + onnxFile);
        }
        Path onnxPath = modelDir.resolve(onnxFile);
        if (!Files.exists(onnxPath)) {
            throw new IOException("Required file missing: " + onnxFile + " (looked in " + modelDir + ")");
        }
        return onnxPath;
    }

    /**
     * Resolve the external data file path, checking both naming conventions.
     *
     * @param modelDir model directory
     * @return path to the external data file
     * @throws IOException if neither file exists
     */
    public static Path resolveExternalDataPath(Path modelDir) throws IOException {
        Path primary = modelDir.resolve(DATA_FILE_PRIMARY);
        if (Files.exists(primary)) return primary;
        Path alt = modelDir.resolve(DATA_FILE_ALT);
        if (Files.exists(alt)) return alt;
        throw new IOException("Required file missing: " + DATA_FILE_PRIMARY
                + " (or " + DATA_FILE_ALT + ") (looked in " + modelDir + ")");
    }

    /**
     * Read the active ONNX filename. Falls back to {@code model.onnx} when unset.
     *
     * @param modelDir model directory
     * @return safe ONNX filename or {@code model.onnx}
     */
    public static String selectedOnnxFilename(Path modelDir) {
        String propertyValue = System.getProperty(ONNX_FILE_PROPERTY);
        if (isSafeOnnxFilename(propertyValue)) {
            return propertyValue.trim();
        }
        if (modelDir == null) {
            return DEFAULT_ONNX_FILE;
        }
        Path selectionFile = modelDir.resolve(SELECTED_ONNX_FILE);
        if (!Files.exists(selectionFile)) {
            return DEFAULT_ONNX_FILE;
        }
        try {
            String value = new String(Files.readAllBytes(selectionFile), StandardCharsets.UTF_8).trim();
            if (isSafeOnnxFilename(value)) {
                return value;
            }
        } catch (IOException ignored) {
            // Use the legacy default when the selection file cannot be read.
        }
        return DEFAULT_ONNX_FILE;
    }

    /**
     * Persist the active ONNX filename used by the runtime loader.
     *
     * @param modelDir     model directory
     * @param onnxFilename safe Hugging Face ONNX filename
     * @throws IOException if the filename is unsafe or cannot be written
     */
    public static void writeSelectedOnnxFilename(Path modelDir, String onnxFilename) throws IOException {
        if (modelDir == null) {
            throw new IOException("Qwen model directory is null");
        }
        if (!isSafeOnnxFilename(onnxFilename)) {
            throw new IOException("Invalid Qwen ONNX filename: " + onnxFilename);
        }
        Files.createDirectories(modelDir);
        Files.write(modelDir.resolve(SELECTED_ONNX_FILE),
                Collections.singletonList(onnxFilename.trim()), StandardCharsets.UTF_8);
    }

    private static boolean isSafeOnnxFilename(String filename) {
        if (filename == null) {
            return false;
        }
        String value = filename.trim();
        return !value.isEmpty()
                && value.endsWith(".onnx")
                && value.indexOf('/') < 0
                && value.indexOf('\\') < 0
                && !value.contains("..");
    }
}
