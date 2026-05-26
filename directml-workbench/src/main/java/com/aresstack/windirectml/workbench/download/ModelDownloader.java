package com.aresstack.windirectml.workbench.download;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.List;
import java.util.function.Consumer;

/**
 * Downloads model files from Hugging Face into a local directory.
 * <p>
 * For embedding models: {@code model.safetensors}, {@code tokenizer.json}, {@code config.json}.
 * For ONNX/decoder models (e.g. Phi-3): {@code model.onnx}, {@code model.onnx.data},
 * {@code tokenizer.json}, {@code config.json}.
 */
public final class ModelDownloader {

    private static final List<String> REQUIRED_FILES = List.of(
            "model.safetensors", "tokenizer.json", "config.json"
    );

    private static final List<String> OPTIONAL_FILES = List.of(
            "vocab.txt", "special_tokens_map.json", "tokenizer_config.json"
    );

    /** Required files for Phi-3 ONNX decoder model. */
    public static final List<String> PHI3_REQUIRED_FILES = List.of(
            "model.onnx", "model.onnx.data", "tokenizer.json", "config.json"
    );

    /** Subdirectory within the HuggingFace repo for Phi-3 DirectML INT4 quantised variant. */
    public static final String PHI3_SUBDIR = "directml-int4-awq-block-128";

    private static final String HF_BASE_URL = "https://huggingface.co";

    private ModelDownloader() {}

    /**
     * Download embedding model files from Hugging Face (safetensors layout).
     *
     * @param repo      HuggingFace repository (e.g. "sentence-transformers/all-MiniLM-L6-v2")
     * @param targetDir local directory to save files into
     * @param force     if true, overwrite existing files
     * @param logger    callback for progress messages
     */
    public static void download(String repo, Path targetDir, boolean force, Consumer<String> logger)
            throws IOException, InterruptedException {
        download(repo, null, REQUIRED_FILES, OPTIONAL_FILES, targetDir, force, logger);
    }

    /**
     * Download model files from Hugging Face with custom file manifests.
     * <p>
     * Use this for decoder/ONNX models that require a different set of files
     * and may store them in a subdirectory within the HuggingFace repository.
     *
     * @param repo          HuggingFace repository (e.g. "microsoft/Phi-3-mini-4k-instruct-onnx")
     * @param repoSubdir    subdirectory within the HuggingFace repo (e.g. "directml-int4-awq-block-128"),
     *                      or {@code null} for root-level files
     * @param requiredFiles list of file names that must be downloaded successfully
     * @param optionalFiles list of file names to attempt downloading (may be absent)
     * @param targetDir     local directory to save files into
     * @param force         if true, overwrite existing files
     * @param logger        callback for progress messages
     */
    public static void download(String repo, String repoSubdir,
                                List<String> requiredFiles, List<String> optionalFiles,
                                Path targetDir, boolean force, Consumer<String> logger)
            throws IOException, InterruptedException {
        Files.createDirectories(targetDir);

        var client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();

        String pathPrefix = repoSubdir != null ? repoSubdir + "/" : "";

        for (String file : requiredFiles) {
            downloadFile(client, repo, pathPrefix + file, file, targetDir, force, logger, true);
        }

        if (optionalFiles != null) {
            for (String file : optionalFiles) {
                downloadFile(client, repo, pathPrefix + file, file, targetDir, force, logger, false);
            }
        }
    }

    /**
     * Download Phi-3 model from HuggingFace (ONNX/GenAI layout).
     * <p>
     * Downloads from the {@code directml-int4-awq-block-128} subdirectory of the
     * {@code microsoft/Phi-3-mini-4k-instruct-onnx} repository.
     *
     * @param targetDir local directory to save model files into
     * @param force     if true, overwrite existing files
     * @param logger    callback for progress messages
     */
    public static void downloadPhi3(Path targetDir, boolean force, Consumer<String> logger)
            throws IOException, InterruptedException {
        download("microsoft/Phi-3-mini-4k-instruct-onnx", PHI3_SUBDIR,
                PHI3_REQUIRED_FILES, List.of(), targetDir, force, logger);
    }

    private static void downloadFile(HttpClient client, String repo,
                                     String remotePath, String localFilename,
                                     Path targetDir, boolean force,
                                     Consumer<String> logger, boolean required)
            throws IOException, InterruptedException {

        Path target = targetDir.resolve(localFilename);

        if (Files.exists(target) && !force) {
            logger.accept("  Skipping (exists): " + localFilename);
            return;
        }

        String url = HF_BASE_URL + "/" + repo + "/resolve/main/" + remotePath;
        logger.accept("  Downloading: " + localFilename + " ...");

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofMinutes(30))
                .GET()
                .build();

        HttpResponse<InputStream> response = client.send(request,
                HttpResponse.BodyHandlers.ofInputStream());

        if (response.statusCode() == 200) {
            try (InputStream in = response.body()) {
                Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
            }
            long sizeKb = Files.size(target) / 1024;
            logger.accept("  Downloaded: " + localFilename + " (" + sizeKb + " KB)");
        } else if (response.statusCode() == 404 && !required) {
            logger.accept("  Optional file not found (skipped): " + localFilename);
        } else {
            String msg = "HTTP " + response.statusCode() + " for " + localFilename
                    + " (url: " + url + ")";
            if (required) {
                throw new IOException(msg);
            } else {
                logger.accept("  Warning: " + msg);
            }
        }
    }
}
