package com.aresstack.windirectml.workbench.download;

import java.util.List;

/**
 * Configuration for downloading a Qwen ONNX model from Hugging Face.
 *
 * <p>Encapsulates the repository layout so the downloader does not hard-code
 * remote paths or filenames. Users can override via the Workbench settings submenu
 * or construct a custom config for testing alternative repos.</p>
 *
 * @param repo            Hugging Face repository identifier (e.g. "onnx-community/Qwen2.5-Coder-0.5B-Instruct")
 * @param onnxSubdir      subdirectory in the repo containing model.onnx and external data (e.g. "onnx")
 * @param modelFile       ONNX graph filename in the remote subdir (e.g. "model.onnx")
 * @param externalDataFile external data filename in the remote subdir (e.g. "model.onnx.data")
 * @param localModelFile  local filename for the ONNX graph (e.g. "model.onnx")
 * @param localDataFile   local filename for the external data (e.g. "model.onnx.data")
 * @param rootFiles       config/tokenizer files at the repo root that are required
 * @param optionalFiles   optional files at the repo root (download is best-effort)
 * @param localDirName    local directory name under the model root
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

    /** Default config for onnx-community/Qwen2.5-Coder-0.5B-Instruct (normal ONNX pair). */
    public static final QwenModelDownloadConfig DEFAULT = new QwenModelDownloadConfig(
            "onnx-community/Qwen2.5-Coder-0.5B-Instruct",
            "onnx",
            "model.onnx",
            "model.onnx.data",
            "model.onnx",
            "model.onnx.data",
            List.of("tokenizer.json", "config.json", "tokenizer_config.json", "special_tokens_map.json"),
            List.of("added_tokens.json"),
            "qwen2.5-coder-0.5b-directml-int4"
    );

    /**
     * Returns the list of local file names that the engine requires after download.
     * This list is used for validation and UI display.
     */
    public List<String> requiredLocalFiles() {
        var files = new java.util.ArrayList<>(rootFiles);
        files.add(localModelFile);
        files.add(localDataFile);
        return List.copyOf(files);
    }

    /**
     * Returns the remote path for the ONNX model file.
     */
    public String remoteModelPath() {
        return onnxSubdir + "/" + modelFile;
    }

    /**
     * Returns the remote path for the external data file.
     */
    public String remoteDataPath() {
        return onnxSubdir + "/" + externalDataFile;
    }
}
