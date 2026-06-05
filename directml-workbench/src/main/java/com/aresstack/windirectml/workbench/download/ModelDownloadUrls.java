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
    public static final String QWEN_SAFETENSORS_REPO = "Qwen/Qwen2.5-Coder-0.5B-Instruct";
    public static final String QWEN_SAFETENSORS_LOCAL_DIR = "qwen2.5-coder-0.5b-safetensors";

    private ModelDownloadUrls() {
    }

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
        ArrayList<ModelFileDescriptor> descriptors = new ArrayList<ModelFileDescriptor>();
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
        ArrayList<ModelFileDescriptor> descriptors = new ArrayList<ModelFileDescriptor>();
        for (String file : ModelDownloader.PHI3_REQUIRED_FILES) {
            String url = buildUrl(repo, subdir + "/" + file);
            descriptors.add(new ModelFileDescriptor(file, true, url, url, file));
        }
        return new ModelDownloadManifest(folder, folder, List.copyOf(descriptors));
    }

    /**
     * Creates a complete download manifest for the selected Qwen ONNX variant.
     */
    public static ModelDownloadManifest manifestForQwen(QwenModelDownloadConfig config) {
        ArrayList<ModelFileDescriptor> descriptors = new ArrayList<ModelFileDescriptor>();
        addQwenModelDescriptors(descriptors, config);
        addQwenRootDescriptors(descriptors, config);
        return new ModelDownloadManifest(config.localDirName(), config.localDirName(), List.copyOf(descriptors));
    }

    /**
     * Creates a complete download manifest for the canonical Qwen dense SafeTensors repository.
     */
    public static ModelDownloadManifest manifestForQwenSafeTensors() {
        ArrayList<ModelFileDescriptor> descriptors = new ArrayList<ModelFileDescriptor>();
        addRootDescriptor(descriptors, QWEN_SAFETENSORS_REPO, "model.safetensors", true);
        addRootDescriptor(descriptors, QWEN_SAFETENSORS_REPO, "config.json", true);
        addRootDescriptor(descriptors, QWEN_SAFETENSORS_REPO, "tokenizer.json", true);
        addRootDescriptor(descriptors, QWEN_SAFETENSORS_REPO, "tokenizer_config.json", true);
        addRootDescriptor(descriptors, QWEN_SAFETENSORS_REPO, "special_tokens_map.json", true);
        addRootDescriptor(descriptors, QWEN_SAFETENSORS_REPO, "generation_config.json", false);
        addRootDescriptor(descriptors, QWEN_SAFETENSORS_REPO, "merges.txt", false);
        addRootDescriptor(descriptors, QWEN_SAFETENSORS_REPO, "vocab.json", false);
        return new ModelDownloadManifest(QWEN_SAFETENSORS_LOCAL_DIR, QWEN_SAFETENSORS_LOCAL_DIR,
                List.copyOf(descriptors));
    }

    /**
     * Creates a small manifest containing only the selected Qwen ONNX model URL.
     */
    public static ModelDownloadManifest modelFileManifestForQwen(QwenModelDownloadConfig config) {
        ArrayList<ModelFileDescriptor> descriptors = new ArrayList<ModelFileDescriptor>();
        addQwenModelDescriptors(descriptors, config);
        return new ModelDownloadManifest(config.localDirName(), config.localDirName(), List.copyOf(descriptors));
    }

    /**
     * Creates a manifest that downloads every known Qwen ONNX file variant.
     */
    public static ModelDownloadManifest manifestForAllQwenVariants() {
        QwenModelDownloadConfig base = QwenModelDownloadConfig.DEFAULT_QUANTIZED;
        ArrayList<ModelFileDescriptor> descriptors = new ArrayList<ModelFileDescriptor>();
        for (QwenOnnxModelVariant variant : QwenOnnxModelVariant.values()) {
            addQwenModelDescriptors(descriptors, QwenModelDownloadConfig.forVariant(variant));
        }
        addQwenRootDescriptors(descriptors, base);
        return new ModelDownloadManifest(base.localDirName(), base.localDirName(), List.copyOf(descriptors));
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
        ArrayList<String> urls = new ArrayList<String>();
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
     */
    public static List<String> forPhi3() {
        String repo = "microsoft/Phi-3-mini-4k-instruct-onnx";
        String subdir = ModelDownloader.PHI3_SUBDIR;
        ArrayList<String> urls = new ArrayList<String>();
        for (String file : ModelDownloader.PHI3_REQUIRED_FILES) {
            urls.add(buildUrl(repo, subdir + "/" + file));
        }
        return List.copyOf(urls);
    }

    /**
     * Returns download URLs for a Qwen model given its configuration.
     */
    public static List<String> forQwen(QwenModelDownloadConfig config) {
        return manifestForQwen(config).files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();
    }

    /**
     * Returns only the selected Qwen ONNX file URL.
     */
    public static String selectedQwenModelUrl(QwenModelDownloadConfig config) {
        return buildUrl(config.repo(), config.remoteModelPath());
    }

    /**
     * Returns the canonical Qwen dense SafeTensors model URL.
     */
    public static String selectedQwenSafeTensorsModelUrl() {
        return buildUrl(QWEN_SAFETENSORS_REPO, "model.safetensors");
    }

    private static void addQwenModelDescriptors(List<ModelFileDescriptor> descriptors,
                                                QwenModelDownloadConfig config) {
        addQwenDescriptor(descriptors, config.modelFile(), true,
                config.remoteModelPath(), config.localModelFile(), config.repo());
        if (config.hasExternalDataFile()) {
            addQwenDescriptor(descriptors, config.externalDataFile(), true,
                    config.remoteDataPath(), config.localDataFile(), config.repo());
        }
    }

    private static void addQwenRootDescriptors(List<ModelFileDescriptor> descriptors,
                                               QwenModelDownloadConfig config) {
        for (String file : config.rootFiles()) {
            addQwenDescriptor(descriptors, file, true, file, file, config.repo());
        }
        for (String file : config.optionalFiles()) {
            addQwenDescriptor(descriptors, file, false, file, file, config.repo());
        }
    }

    private static void addRootDescriptor(List<ModelFileDescriptor> descriptors, String repo,
                                          String fileName, boolean required) {
        String url = buildUrl(repo, fileName);
        descriptors.add(new ModelFileDescriptor(fileName, required, url, url, fileName));
    }

    private static void addQwenDescriptor(List<ModelFileDescriptor> descriptors, String displayName,
                                          boolean required, String remotePath, String localFilename,
                                          String repo) {
        String url = buildUrl(repo, remotePath);
        descriptors.add(new ModelFileDescriptor(displayName, required, url, url, localFilename));
    }

    private static String buildUrl(String repo, String remotePath) {
        return HF_BASE_URL + "/" + repo + "/resolve/main/" + remotePath;
    }
}
