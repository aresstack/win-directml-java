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
    public static final String QWEN_CODER_1_5B_REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct";
    public static final String QWEN_CODER_1_5B_LOCAL_DIR = "qwen2.5-coder-1.5b-instruct";
    public static final String QWEN_CODER_3B_REPO = "Qwen/Qwen2.5-Coder-3B-Instruct";
    public static final String QWEN_CODER_3B_LOCAL_DIR = "qwen2.5-coder-3b-instruct";

    public static final String SMOLLM2_135M_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct";
    public static final String SMOLLM2_135M_LOCAL_DIR = "smollm2-135m-instruct";
    public static final String SMOLLM2_360M_REPO = "HuggingFaceTB/SmolLM2-360M-Instruct";
    public static final String SMOLLM2_360M_LOCAL_DIR = "smollm2-360m-instruct";
    public static final String CODET5_SMALL_REPO = "Salesforce/codet5-small";
    public static final String CODET5_SMALL_LOCAL_DIR = "codet5-small";
    public static final String CODET5_BASE_MULTI_SUM_REPO = "Salesforce/codet5-base-multi-sum";
    public static final String CODET5_BASE_MULTI_SUM_LOCAL_DIR = "codet5-base-multi-sum";
    public static final String GOOGLE_T5_SMALL_REPO = "google-t5/t5-small";
    public static final String GOOGLE_T5_SMALL_LOCAL_DIR = "t5-small";
    public static final String GOOGLE_FLAN_T5_SMALL_REPO = "google/flan-t5-small";
    public static final String GOOGLE_FLAN_T5_SMALL_LOCAL_DIR = "flan-t5-small";
    public static final String GEMMA3_270M_REPO = "google/gemma-3-270m";
    public static final String GEMMA3_270M_LOCAL_DIR = "gemma-3-270m";
    public static final String GEMMA3_270M_IT_REPO = "google/gemma-3-270m-it";
    public static final String GEMMA3_270M_IT_LOCAL_DIR = "gemma-3-270m-it";

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
     * Creates a complete download manifest for Qwen2.5-Coder-1.5B-Instruct.
     */
    public static ModelDownloadManifest manifestForQwenCoder1_5BSafeTensors() {
        return manifestForQwenCoderSafeTensors(QWEN_CODER_1_5B_REPO, QWEN_CODER_1_5B_LOCAL_DIR,
                List.of("model.safetensors"));
    }

    /**
     * Creates a complete download manifest for Qwen2.5-Coder-3B-Instruct.
     */
    public static ModelDownloadManifest manifestForQwenCoder3BSafeTensors() {
        return manifestForQwenCoderSafeTensors(QWEN_CODER_3B_REPO, QWEN_CODER_3B_LOCAL_DIR,
                List.of("model-00001-of-00002.safetensors",
                        "model-00002-of-00002.safetensors",
                        "model.safetensors.index.json"));
    }

    /**
     * Creates a complete download manifest for SmolLM2-135M-Instruct.
     */
    public static ModelDownloadManifest manifestForSmolLm2_135M() {
        return manifestForLlamaStyleSafeTensors(SMOLLM2_135M_REPO, SMOLLM2_135M_LOCAL_DIR);
    }

    /**
     * Creates a complete download manifest for SmolLM2-360M-Instruct.
     */
    public static ModelDownloadManifest manifestForSmolLm2_360M() {
        return manifestForLlamaStyleSafeTensors(SMOLLM2_360M_REPO, SMOLLM2_360M_LOCAL_DIR);
    }

    /**
     * Creates a complete download manifest for Gemma 3 270M.
     */
    public static ModelDownloadManifest manifestForGemma3_270M() {
        return manifestForGemma3SafeTensors(GEMMA3_270M_REPO, GEMMA3_270M_LOCAL_DIR, false);
    }

    /**
     * Creates a complete download manifest for Gemma 3 270M Instruct.
     */
    public static ModelDownloadManifest manifestForGemma3_270MInstruct() {
        return manifestForGemma3SafeTensors(GEMMA3_270M_IT_REPO, GEMMA3_270M_IT_LOCAL_DIR, true);
    }

    /**
     * Creates a complete download manifest for Salesforce CodeT5-small.
     *
     * <p>The upstream repository publishes its weights as {@code pytorch_model.bin}. The file is
     * downloaded only as an import source for the restricted Torch state-dict compiler and is never
     * used by the runtime directly.</p>
     */
    public static ModelDownloadManifest manifestForCodeT5Small() {
        return manifestForCodeT5TorchCheckpoint(CODET5_SMALL_REPO, CODET5_SMALL_LOCAL_DIR);
    }

    /**
     * Creates a complete download manifest for Salesforce CodeT5 base multi-language summarization.
     */
    public static ModelDownloadManifest manifestForCodeT5BaseMultiSum() {
        return manifestForCodeT5TorchCheckpoint(CODET5_BASE_MULTI_SUM_REPO, CODET5_BASE_MULTI_SUM_LOCAL_DIR);
    }

    private static ModelDownloadManifest manifestForCodeT5TorchCheckpoint(String repo, String localDir) {
        ArrayList<ModelFileDescriptor> descriptors = new ArrayList<ModelFileDescriptor>();
        addRootDescriptor(descriptors, repo, "pytorch_model.bin", true);
        addRootDescriptor(descriptors, repo, "config.json", true);
        addRootDescriptor(descriptors, repo, "vocab.json", true);
        addRootDescriptor(descriptors, repo, "merges.txt", true);
        addRootDescriptor(descriptors, repo, "tokenizer_config.json", true);
        addRootDescriptor(descriptors, repo, "special_tokens_map.json", true);
        addRootDescriptor(descriptors, repo, "added_tokens.json", false);
        return new ModelDownloadManifest(repo, localDir, List.copyOf(descriptors));
    }

    /**
     * Creates a complete download manifest for google-t5/t5-small.
     *
     * <p>This model is useful as the first upstream T5 SafeTensors validation model:
     * the repository publishes {@code model.safetensors} directly and therefore does
     * not require a PyTorch checkpoint conversion step.</p>
     */
    public static ModelDownloadManifest manifestForGoogleT5Small() {
        ArrayList<ModelFileDescriptor> descriptors = new ArrayList<ModelFileDescriptor>();
        addRootDescriptor(descriptors, GOOGLE_T5_SMALL_REPO, "model.safetensors", true);
        addRootDescriptor(descriptors, GOOGLE_T5_SMALL_REPO, "config.json", true);
        addRootDescriptor(descriptors, GOOGLE_T5_SMALL_REPO, "tokenizer.json", true);
        addRootDescriptor(descriptors, GOOGLE_T5_SMALL_REPO, "tokenizer_config.json", true);
        addRootDescriptor(descriptors, GOOGLE_T5_SMALL_REPO, "spiece.model", true);
        addRootDescriptor(descriptors, GOOGLE_T5_SMALL_REPO, "generation_config.json", false);
        return new ModelDownloadManifest(GOOGLE_T5_SMALL_REPO, GOOGLE_T5_SMALL_LOCAL_DIR, List.copyOf(descriptors));
    }

    /**
     * Creates a complete download manifest for google/flan-t5-small.
     *
     * <p>This is the preferred upstream T5-family summarizer smoke model because it
     * is instruction-tuned and publishes {@code model.safetensors} directly.</p>
     */
    public static ModelDownloadManifest manifestForGoogleFlanT5Small() {
        ArrayList<ModelFileDescriptor> descriptors = new ArrayList<ModelFileDescriptor>();
        addRootDescriptor(descriptors, GOOGLE_FLAN_T5_SMALL_REPO, "model.safetensors", true);
        addRootDescriptor(descriptors, GOOGLE_FLAN_T5_SMALL_REPO, "config.json", true);
        addRootDescriptor(descriptors, GOOGLE_FLAN_T5_SMALL_REPO, "tokenizer.json", true);
        addRootDescriptor(descriptors, GOOGLE_FLAN_T5_SMALL_REPO, "tokenizer_config.json", true);
        addRootDescriptor(descriptors, GOOGLE_FLAN_T5_SMALL_REPO, "spiece.model", true);
        addRootDescriptor(descriptors, GOOGLE_FLAN_T5_SMALL_REPO, "special_tokens_map.json", false);
        addRootDescriptor(descriptors, GOOGLE_FLAN_T5_SMALL_REPO, "generation_config.json", false);
        return new ModelDownloadManifest(GOOGLE_FLAN_T5_SMALL_REPO, GOOGLE_FLAN_T5_SMALL_LOCAL_DIR,
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
     * Returns download URLs for Qwen2.5-Coder-1.5B-Instruct.
     */
    public static List<String> forQwenCoder1_5BSafeTensors() {
        return manifestForQwenCoder1_5BSafeTensors().files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();
    }

    /**
     * Returns download URLs for Qwen2.5-Coder-3B-Instruct.
     */
    public static List<String> forQwenCoder3BSafeTensors() {
        return manifestForQwenCoder3BSafeTensors().files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();
    }

    /**
     * Returns download URLs for SmolLM2-135M-Instruct.
     */
    public static List<String> forSmolLm2_135M() {
        return manifestForSmolLm2_135M().files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();
    }

    /**
     * Returns download URLs for SmolLM2-360M-Instruct.
     */
    public static List<String> forSmolLm2_360M() {
        return manifestForSmolLm2_360M().files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();
    }

    /**
     * Returns download URLs for Gemma 3 270M.
     */
    public static List<String> forGemma3_270M() {
        return manifestForGemma3_270M().files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();
    }

    /**
     * Returns download URLs for Gemma 3 270M Instruct.
     */
    public static List<String> forGemma3_270MInstruct() {
        return manifestForGemma3_270MInstruct().files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();
    }

    /**
     * Returns download URLs for CodeT5-small.
     */
    public static List<String> forCodeT5Small() {
        return manifestForCodeT5Small().files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();
    }

    /**
     * Returns download URLs for CodeT5 base multi-language summarization.
     */
    public static List<String> forCodeT5BaseMultiSum() {
        return manifestForCodeT5BaseMultiSum().files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();
    }


    /**
     * Returns download URLs for google-t5/t5-small.
     */
    public static List<String> forGoogleT5Small() {
        return manifestForGoogleT5Small().files().stream()
                .map(ModelFileDescriptor::defaultUrl)
                .toList();
    }

    /**
     * Returns download URLs for google/flan-t5-small.
     */
    public static List<String> forGoogleFlanT5Small() {
        return manifestForGoogleFlanT5Small().files().stream()
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

    private static ModelDownloadManifest manifestForGemma3SafeTensors(String repo, String localDirName,
                                                                   boolean instructionTuned) {
        ArrayList<ModelFileDescriptor> descriptors = new ArrayList<ModelFileDescriptor>();
        addRootDescriptor(descriptors, repo, "model.safetensors", true);
        addRootDescriptor(descriptors, repo, "config.json", true);
        addRootDescriptor(descriptors, repo, "tokenizer.json", true);
        addRootDescriptor(descriptors, repo, "tokenizer.model", true);
        addRootDescriptor(descriptors, repo, "tokenizer_config.json", true);
        addRootDescriptor(descriptors, repo, "special_tokens_map.json", true);
        addRootDescriptor(descriptors, repo, "added_tokens.json", false);
        addRootDescriptor(descriptors, repo, "generation_config.json", false);
        if (instructionTuned) {
            addRootDescriptor(descriptors, repo, "chat_template.jinja", false);
        }
        return new ModelDownloadManifest(repo, localDirName, List.copyOf(descriptors));
    }

    private static ModelDownloadManifest manifestForQwenCoderSafeTensors(String repo, String localDirName,
                                                                         List<String> weightFiles) {
        ArrayList<ModelFileDescriptor> descriptors = new ArrayList<ModelFileDescriptor>();
        for (String file : weightFiles) {
            addRootDescriptor(descriptors, repo, file, true);
        }
        addRootDescriptor(descriptors, repo, "config.json", true);
        addRootDescriptor(descriptors, repo, "tokenizer.json", true);
        addRootDescriptor(descriptors, repo, "tokenizer_config.json", true);
        addRootDescriptor(descriptors, repo, "generation_config.json", true);
        addRootDescriptor(descriptors, repo, "merges.txt", true);
        addRootDescriptor(descriptors, repo, "vocab.json", true);
        return new ModelDownloadManifest(repo, localDirName, List.copyOf(descriptors));
    }

    private static ModelDownloadManifest manifestForLlamaStyleSafeTensors(String repo, String localDirName) {
        ArrayList<ModelFileDescriptor> descriptors = new ArrayList<ModelFileDescriptor>();
        addRootDescriptor(descriptors, repo, "model.safetensors", true);
        addRootDescriptor(descriptors, repo, "config.json", true);
        addRootDescriptor(descriptors, repo, "tokenizer.json", true);
        addRootDescriptor(descriptors, repo, "tokenizer_config.json", true);
        addRootDescriptor(descriptors, repo, "special_tokens_map.json", true);
        addRootDescriptor(descriptors, repo, "generation_config.json", false);
        addRootDescriptor(descriptors, repo, "merges.txt", false);
        addRootDescriptor(descriptors, repo, "vocab.json", false);
        return new ModelDownloadManifest(repo, localDirName, List.copyOf(descriptors));
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
