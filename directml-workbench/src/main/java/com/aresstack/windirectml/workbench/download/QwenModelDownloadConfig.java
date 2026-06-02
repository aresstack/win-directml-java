package com.aresstack.windirectml.workbench.download;

import java.util.List;

/**
 * Configuration for downloading a Qwen ONNX model from Hugging Face.
 *
 * <p>Encapsulates the repository layout so the downloader does not hard-code
 * remote paths or filenames. The external-data entries may be blank for
 * single-file ONNX artifacts such as {@code model_q4f16.onnx}.</p>
 *
 * @param repo             Hugging Face repository identifier
 * @param onnxSubdir       subdirectory in the repo containing ONNX files
 * @param modelFile        ONNX model filename in the remote subdir
 * @param externalDataFile external data filename in the remote subdir, or blank for single-file ONNX
 * @param localModelFile   local filename for the ONNX graph
 * @param localDataFile    local filename for the external data, or blank for single-file ONNX
 * @param rootFiles        config/tokenizer files at the repo root that are required
 * @param optionalFiles    optional files at the repo root (download is best-effort)
 * @param localDirName     local directory name under the model root
 */
public record QwenModelDownloadConfig(
        String repo,
        String onnxSubdir,
        String modelFile,
        String externalDataFile,
        String localModelFile,
        String localDataFile,
        List<String> rootFiles,
        List<String> optionalFiles,
        String localDirName
) {

    /**
     * Default config for the ONNX Community q4f16 single-file artifact.
     */
    public static final QwenModelDownloadConfig DEFAULT = new QwenModelDownloadConfig(
            "onnx-community/Qwen2.5-Coder-0.5B-Instruct",
            "onnx",
            "model_q4f16.onnx",
            "",
            "model.onnx",
            "",
            List.of("tokenizer.json", "config.json", "tokenizer_config.json", "special_tokens_map.json"),
            List.of("added_tokens.json", "generation_config.json"),
            "qwen2.5-coder-0.5b-directml-int4"
    );

    /**
     * Returns whether this configuration downloads a separate ONNX external-data sidecar.
     *
     * @return {@code true} if both remote and local sidecar filenames are configured
     */
    public boolean hasExternalDataFile() {
        return hasText(externalDataFile) && hasText(localDataFile);
    }

    /**
     * Returns the list of local file names that the engine requires after download.
     * This list is used for validation and UI display.
     */
    public List<String> requiredLocalFiles() {
        var files = new java.util.ArrayList<String>(rootFiles);
        files.add(localModelFile);
        if (hasExternalDataFile()) {
            files.add(localDataFile);
        }
        return List.copyOf(files);
    }

    /**
     * Returns the remote path for the ONNX model file.
     */
    public String remoteModelPath() {
        return onnxSubdir + "/" + modelFile;
    }

    /**
     * Returns the remote path for the external data file, or an empty string for single-file ONNX.
     */
    public String remoteDataPath() {
        if (!hasExternalDataFile()) {
            return "";
        }
        return onnxSubdir + "/" + externalDataFile;
    }

    private static boolean hasText(String value) {
        return value != null && !value.trim().isEmpty();
    }
}
