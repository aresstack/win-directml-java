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
 * Required files: {@code model.safetensors}, {@code tokenizer.json}, {@code config.json}.
 * Optional files: {@code vocab.txt}, {@code special_tokens_map.json}, {@code tokenizer_config.json}.
 */
public final class ModelDownloader {

    private static final List<String> REQUIRED_FILES = List.of(
            "model.safetensors", "tokenizer.json", "config.json"
    );

    private static final List<String> OPTIONAL_FILES = List.of(
            "vocab.txt", "special_tokens_map.json", "tokenizer_config.json"
    );

    private static final String HF_BASE_URL = "https://huggingface.co";

    private ModelDownloader() {}

    /**
     * Download model files from Hugging Face.
     *
     * @param repo      HuggingFace repository (e.g. "sentence-transformers/all-MiniLM-L6-v2")
     * @param targetDir local directory to save files into
     * @param force     if true, overwrite existing files
     * @param logger    callback for progress messages
     */
    public static void download(String repo, Path targetDir, boolean force, Consumer<String> logger)
            throws IOException, InterruptedException {
        Files.createDirectories(targetDir);

        var client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();

        for (String file : REQUIRED_FILES) {
            downloadFile(client, repo, file, targetDir, force, logger, true);
        }

        for (String file : OPTIONAL_FILES) {
            downloadFile(client, repo, file, targetDir, force, logger, false);
        }
    }

    private static void downloadFile(HttpClient client, String repo, String filename,
                                     Path targetDir, boolean force,
                                     Consumer<String> logger, boolean required)
            throws IOException, InterruptedException {

        Path target = targetDir.resolve(filename);

        if (Files.exists(target) && !force) {
            logger.accept("  Skipping (exists): " + filename);
            return;
        }

        String url = HF_BASE_URL + "/" + repo + "/resolve/main/" + filename;
        logger.accept("  Downloading: " + filename + " ...");

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofMinutes(10))
                .GET()
                .build();

        HttpResponse<InputStream> response = client.send(request,
                HttpResponse.BodyHandlers.ofInputStream());

        if (response.statusCode() == 200) {
            try (InputStream in = response.body()) {
                Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
            }
            long sizeKb = Files.size(target) / 1024;
            logger.accept("  Downloaded: " + filename + " (" + sizeKb + " KB)");
        } else if (response.statusCode() == 404 && !required) {
            logger.accept("  Optional file not found (skipped): " + filename);
        } else {
            String msg = "HTTP " + response.statusCode() + " for " + filename;
            if (required) {
                throw new IOException(msg);
            } else {
                logger.accept("  Warning: " + msg);
            }
        }
    }
}
