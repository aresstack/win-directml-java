package com.aresstack.windirectml.workbench.download;

import java.util.ArrayList;
import java.util.List;

/**
 * Computes download URLs for model files without performing any I/O.
 * <p>
 * Used by the UI to expose copy-to-clipboard functionality for model download URLs
 * and to generate {@link ModelDownloadManifest} instances for the configuration dialog.
 */
public final class ModelDownloadUrls {

    private static final String HF_BASE_URL = "https://huggingface.co";

    private ModelDownloadUrls() {}

    // ---- Manifest builders ----

    /**
     * Creates a download manifest for an embedding model.
     */
    public static ModelDownloadManifest manifestForEmbedding(String repo, String folder) {
        List<String> requiredFiles = List.of(
                "model.safetensors", "tokenizer.json", "config.json"
        );
        List<String> optionalFiles = List.of(
                "vocab.txt", "special_tokens_map.json", "tokenizer_config.json"
        );
        var descriptors = new ArrayList<ModelFileDescriptor>();
        for (String file : requiredFiles) {
            String url = buildUrl(repo, file);
            descriptors.add(new ModelFileDescriptor(file, true, url, url, file));
        }
        for (String file : optionalFiles) {
            String url = buildUrl(repo, file);
            descriptors.add(new ModelFileDescriptor(file, false, url, url, file));
        }
        return new ModelDownloadManifest(folder, folder, List.copyOf(descriptors));
    }

    /**
     * Creates a download manifest for the Phi-3 ONNX model.
     */
    public static ModelDownloadManifest manifestForPhi3() {
        String repo = "microsoft/Phi-3-mini-4k-instruct-onnx";
        String subdir = ModelDownloader.PHI3_SUBDIR;
        String folder = "phi-3-mini-4k-instruct-onnx";
        var descriptors = new ArrayList<ModelFileDescriptor>();
        for (String file : ModelDownloader.PHI3_REQUIRED_FILES) {
            String url = buildUrl(repo, subdir + "/" + file);
            descriptors.add(new ModelFileDescriptor(file, true, url, url, file));
        }
        return new ModelDownloadManifest(folder, folder, List.copyOf(descriptors));
    }

    /**
     * Creates a download manifest for a Qwen model given its configuration.
     */
    public static ModelDownloadManifest manifestForQwen(QwenModelDownloadConfig config) {
        var descriptors = new ArrayList<ModelFileDescriptor>();

        String modelUrl = buildUrl(config.repo(), config.remoteModelPath());
        descriptors.add(new ModelFileDescriptor(
                config.localModelFile(), true, modelUrl, modelUrl, config.localModelFile()));

        String dataUrl = buildUrl(config.repo(), config.remoteDataPath());
        descriptors.add(new ModelFileDescriptor(
                config.localDataFile(), true, dataUrl, dataUrl, config.localDataFile()));

        for (String file : config.rootFiles()) {
            String url = buildUrl(config.repo(), file);
            descriptors.add(new ModelFileDescriptor(file, true, url, url, file));
        }
        for (String file : config.optionalFiles()) {
            String url = buildUrl(config.repo(), file);
            descriptors.add(new ModelFileDescriptor(file, false, url, url, file));
        }
        return new ModelDownloadManifest(config.localDirName(), config.localDirName(),
                List.copyOf(descriptors));
    }

    /**
     * Returns download URLs for an embedding model (safetensors layout).
     *
     * @param repo HuggingFace repository (e.g. "sentence-transformers/all-MiniLM-L6-v2")
     * @return list of fully-qualified download URLs
     */
    public static List<String> forEmbeddingModel(String repo) {
        List<String> requiredFiles = List.of(
                "model.safetensors", "tokenizer.json", "config.json"
        );
        List<String> optionalFiles = List.of(
                "vocab.txt", "special_tokens_map.json", "tokenizer_config.json"
        );
        var urls = new ArrayList<String>();
        for (String file : requiredFiles) {
            urls.add(buildUrl(repo, file));
        }
        for (String file : optionalFiles) {
            urls.add(buildUrl(repo, file));
        }
        return List.copyOf(urls);
    }

    /**
     * Returns download URLs for the Phi-3 ONNX/GenAI model.
     *
     * @return list of fully-qualified download URLs
     */
    public static List<String> forPhi3() {
        String repo = "microsoft/Phi-3-mini-4k-instruct-onnx";
        String subdir = ModelDownloader.PHI3_SUBDIR;
        var urls = new ArrayList<String>();
        for (String file : ModelDownloader.PHI3_REQUIRED_FILES) {
            urls.add(buildUrl(repo, subdir + "/" + file));
        }
        return List.copyOf(urls);
    }

    /**
     * Returns download URLs for a Qwen model given its configuration.
     *
     * @param config the Qwen download configuration
     * @return list of fully-qualified download URLs
     */
    public static List<String> forQwen(QwenModelDownloadConfig config) {
        var urls = new ArrayList<String>();
        urls.add(buildUrl(config.repo(), config.remoteModelPath()));
        urls.add(buildUrl(config.repo(), config.remoteDataPath()));
        for (String file : config.rootFiles()) {
            urls.add(buildUrl(config.repo(), file));
        }
        for (String file : config.optionalFiles()) {
            urls.add(buildUrl(config.repo(), file));
        }
        return List.copyOf(urls);
    }

    private static String buildUrl(String repo, String remotePath) {
        return HF_BASE_URL + "/" + repo + "/resolve/main/" + remotePath;
    }
}
