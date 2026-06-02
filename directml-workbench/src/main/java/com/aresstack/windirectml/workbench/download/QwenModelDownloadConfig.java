package com.aresstack.windirectml.workbench.download;

import java.util.List;

/**
 * Configuration for downloading a Qwen ONNX model from Hugging Face.
 *
 * <p>Encapsulates the repository layout so the downloader does not hard-code
 * remote paths or filenames. The Workbench uses the same config to download
 * the selected ONNX file and to show the selected model URL.</p>
 *
 * @param repo             Hugging Face repository identifier
 * @param onnxSubdir       subdirectory in the repo containing ONNX files
 * @param modelFile        ONNX graph/weights filename in the remote subdir
 * @param externalDataFile optional external data filename in the remote subdir
 * @param localModelFile   local ONNX filename; this intentionally keeps the Hugging Face filename
 * @param localDataFile    optional local external-data filename
 * @param rootFiles        config/tokenizer files at the repo root that are required
 * @param optionalFiles    optional files at the repo root
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

    public static final String REPO = "onnx-community/Qwen2.5-Coder-0.5B-Instruct";
    public static final String ONNX_SUBDIR = "onnx";
    public static final String LOCAL_DIR_NAME = "qwen2.5-coder-0.5b-directml-int4";

    /**
     * Default config kept for compatibility with the original dense ONNX pair.
     */
    public static final QwenModelDownloadConfig DEFAULT = forVariant(QwenOnnxModelVariant.DEFAULT_DENSE);

    /**
     * Workbench default for the INT4 test path.
     */
    public static final QwenModelDownloadConfig DEFAULT_QUANTIZED = forVariant(QwenOnnxModelVariant.Q4F16);

    /**
     * Create a config for one selectable Hugging Face ONNX filename.
     */
    public static QwenModelDownloadConfig forVariant(QwenOnnxModelVariant variant) {
        return new QwenModelDownloadConfig(
                REPO,
                ONNX_SUBDIR,
                variant.modelFileName(),
                variant.externalDataFileName(),
                variant.modelFileName(),
                variant.externalDataFileName(),
                List.of("tokenizer.json", "config.json", "tokenizer_config.json", "special_tokens_map.json"),
                List.of("added_tokens.json", "generation_config.json"),
                LOCAL_DIR_NAME
        );
    }

    /**
     * Create a config from a selected local/Hugging Face ONNX filename.
     */
    public static QwenModelDownloadConfig forModelFile(String modelFileName) {
        return forVariant(QwenOnnxModelVariant.fromModelFileName(modelFileName));
    }

    /**
     * Returns the list of local file names that the engine requires after download.
     * This list is used for validation and UI display.
     */
    public List<String> requiredLocalFiles() {
        java.util.ArrayList<String> files = new java.util.ArrayList<String>(rootFiles);
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
     * Returns the remote path for the external data file.
     */
    public String remoteDataPath() {
        if (!hasExternalDataFile()) {
            return null;
        }
        return onnxSubdir + "/" + externalDataFile;
    }

    public boolean hasExternalDataFile() {
        return externalDataFile != null && !externalDataFile.isBlank()
                && localDataFile != null && !localDataFile.isBlank();
    }
}
