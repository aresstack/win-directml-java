package com.aresstack.windirectml.workbench.download;

import com.aresstack.winproxy.ProxyConfiguration;
import com.aresstack.winproxy.ProxyResult;
import com.aresstack.winproxy.WindowsProxyResolver;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Proxy;
import java.net.ProxySelector;
import java.net.SocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.Collections;
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

    private static final int DOWNLOAD_BUFFER_SIZE = 1024 * 1024;

    /**
     * Required files for Phi-3 ONNX decoder model.
     */
    public static final List<String> PHI3_REQUIRED_FILES = List.of(
            "model.onnx", "model.onnx.data", "tokenizer.json", "config.json"
    );

    /**
     * Subdirectory within the HuggingFace repo for Phi-3 DirectML INT4 quantised variant.
     */
    public static final String PHI3_SUBDIR = "directml/directml-int4-awq-block-128";

    /**
     * Required local files for Qwen2.5-Coder ONNX decoder model (derived from DEFAULT config).
     */
    public static final List<String> QWEN_REQUIRED_FILES =
            QwenModelDownloadConfig.DEFAULT.requiredLocalFiles();

    /**
     * Subdirectory within the HuggingFace repo for onnx-community Qwen2.5-Coder ONNX files.
     */
    public static final String QWEN_ONNX_SUBDIR = QwenModelDownloadConfig.DEFAULT.onnxSubdir();

    private static final String HF_BASE_URL = "https://huggingface.co";

    private ModelDownloader() {
    }

    /**
     * Receives byte-level download progress for the currently processed file.
     */
    @FunctionalInterface
    public interface ProgressListener {
        void onProgress(ProgressEvent event);
    }

    /**
     * Byte-level progress event for one file inside a model manifest.
     */
    public record ProgressEvent(
            String localFilename,
            int fileIndex,
            int totalFiles,
            long bytesRead,
            long totalBytes,
            boolean completed,
            boolean skipped
    ) {
        public double fileFraction() {
            if (skipped || completed) {
                return 1.0d;
            }
            if (totalBytes <= 0L) {
                return 0.0d;
            }
            return Math.max(0.0d, Math.min(1.0d, bytesRead / (double) totalBytes));
        }

        public int aggregatePercent() {
            if (totalFiles <= 0) {
                return 100;
            }
            double completedFiles = Math.max(0, fileIndex) + fileFraction();
            return (int) Math.round(Math.max(0.0d, Math.min(100.0d,
                    (completedFiles * 100.0d) / totalFiles)));
        }
    }

    private static final ProgressListener NO_PROGRESS = event -> {
    };

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
     * @param repoSubdir    subdirectory within the HuggingFace repo (e.g. "directml/directml-int4-awq-block-128"),
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
        int totalFiles = requiredFiles.size() + (optionalFiles == null ? 0 : optionalFiles.size());
        int index = 0;

        for (String file : requiredFiles) {
            String url = HF_BASE_URL + "/" + repo + "/resolve/main/" + pathPrefix + file;
            downloadFileFromUrl(client, url, file, targetDir, force, logger, true,
                    index++, totalFiles, NO_PROGRESS);
        }

        if (optionalFiles != null) {
            for (String file : optionalFiles) {
                String url = HF_BASE_URL + "/" + repo + "/resolve/main/" + pathPrefix + file;
                downloadFileFromUrl(client, url, file, targetDir, force, logger, false,
                        index++, totalFiles, NO_PROGRESS);
            }
        }
    }

    /**
     * Download Phi-3 model from HuggingFace (ONNX/GenAI layout).
     * <p>
     * Downloads from the {@code directml/directml-int4-awq-block-128} subdirectory of the
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

    /**
     * Download Qwen2.5-Coder model files from a HuggingFace ONNX candidate repo
     * using an explicit configuration object.
     *
     * <p>The config determines which remote paths to fetch and what local filenames to use.
     * This allows the Workbench settings submenu to override the default layout.
     *
     * @param config    download configuration (repo, paths, filenames)
     * @param targetDir local directory to save model files into
     * @param force     if true, overwrite existing files
     * @param logger    callback for progress messages
     */
    public static void downloadQwen(QwenModelDownloadConfig config, Path targetDir,
                                    boolean force, Consumer<String> logger)
            throws IOException, InterruptedException {
        downloadFromManifest(ModelDownloadUrls.manifestForQwen(config), targetDir, force, logger);
    }

    /**
     * Download model files using a {@link ModelDownloadManifest}.
     *
     * <p>Each file descriptor in the manifest provides the effective download URL
     * (which may have been overridden by the user) and the local filename.
     *
     * @param manifest  the download manifest with effective URLs
     * @param targetDir local directory to save files into
     * @param force     if true, overwrite existing files
     * @param logger    callback for progress messages
     */
    public static void downloadFromManifest(ModelDownloadManifest manifest, Path targetDir,
                                            boolean force, Consumer<String> logger)
            throws IOException, InterruptedException {
        downloadFromManifest(manifest, targetDir, force, logger, NO_PROGRESS);
    }

    /**
     * Download model files using a {@link ModelDownloadManifest} and report byte progress.
     *
     * @param manifest         the download manifest with effective URLs
     * @param targetDir        local directory to save files into
     * @param force            if true, overwrite existing files
     * @param logger           callback for progress messages
     * @param progressListener byte-level progress callback for each manifest file
     */
    public static void downloadFromManifest(ModelDownloadManifest manifest, Path targetDir,
                                            boolean force, Consumer<String> logger,
                                            ProgressListener progressListener)
            throws IOException, InterruptedException {
        downloadFromManifest(manifest, targetDir, force, logger, progressListener, ProxyConfiguration.defaults());
    }

    /**
     * Download model files using a manifest, byte progress and a workbench proxy configuration.
     *
     * @param manifest           the download manifest with effective URLs
     * @param targetDir          local directory to save files into
     * @param force              if true, overwrite existing files
     * @param logger             callback for progress messages
     * @param progressListener   byte-level progress callback for each manifest file
     * @param proxyConfiguration Windows proxy configuration for outbound HTTP requests
     */
    public static void downloadFromManifest(ModelDownloadManifest manifest, Path targetDir,
                                            boolean force, Consumer<String> logger,
                                            ProgressListener progressListener,
                                            ProxyConfiguration proxyConfiguration)
            throws IOException, InterruptedException {
        Files.createDirectories(targetDir);

        HttpClient client = createHttpClient(proxyConfiguration);

        ProgressListener listener = progressListener == null ? NO_PROGRESS : progressListener;
        int totalFiles = manifest.files().size();
        for (int i = 0; i < totalFiles; i++) {
            ModelFileDescriptor desc = manifest.files().get(i);
            downloadFileFromUrl(client, desc.currentUrl(), desc.localFilename(),
                    targetDir, force, logger, desc.required(), i, totalFiles, listener);
        }
    }

    /**
     * Returns the required local file names declared by {@code manifest} that are
     * absent or zero-byte under {@code targetDir}.
     *
     * <p>Used after a download to detect an interrupted / partial install: an
     * empty result means every required artefact is present and non-empty.
     * Optional files are ignored.</p>
     *
     * @param manifest  the model download manifest
     * @param targetDir the local directory the model was downloaded into
     * @return the missing or zero-byte required local filenames (never {@code null})
     */
    public static List<String> missingRequiredFiles(ModelDownloadManifest manifest, Path targetDir) {
        java.util.ArrayList<String> missing = new java.util.ArrayList<String>();
        if (manifest == null || targetDir == null) {
            return missing;
        }
        for (ModelFileDescriptor descriptor : manifest.files()) {
            if (!descriptor.required()) {
                continue;
            }
            Path file = targetDir.resolve(descriptor.localFilename());
            try {
                if (!Files.isRegularFile(file) || Files.size(file) == 0L) {
                    missing.add(descriptor.localFilename());
                }
            } catch (IOException e) {
                missing.add(descriptor.localFilename());
            }
        }
        return missing;
    }

    private static HttpClient createHttpClient(ProxyConfiguration proxyConfiguration) {
        ProxyConfiguration configuration = proxyConfiguration == null
                ? ProxyConfiguration.defaults() : proxyConfiguration;
        WindowsProxyResolver resolver = new WindowsProxyResolver(configuration);
        return HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .proxy(new ProxySelector() {
                    @Override
                    public List<Proxy> select(URI uri) {
                        try {
                            ProxyResult result = resolver.resolve(uri.toString());
                            if (result == null || !result.isProxy()) {
                                // DIRECT, ERROR or NOT_IMPLEMENTED -> best-effort direct connection.
                                return Collections.singletonList(Proxy.NO_PROXY);
                            }
                            return Collections.singletonList(new Proxy(Proxy.Type.HTTP,
                                    InetSocketAddress.createUnresolved(result.getHost(), result.getPort())));
                        } catch (RuntimeException ex) {
                            throw new IllegalStateException("Proxy resolution failed for " + uri + ": " + describe(ex), ex);
                        }
                    }

                    @Override
                    public void connectFailed(URI uri, SocketAddress sa, IOException ioe) {
                        // Best-effort proxy resolution for downloads; the download request reports failures.
                    }
                })
                .build();
    }

    private static String describe(Throwable throwable) {
        if (throwable == null) {
            return "unknown error";
        }
        String message = throwable.getMessage();
        if (message == null || message.trim().isEmpty()) {
            return throwable.getClass().getName();
        }
        return throwable.getClass().getName() + ": " + message;
    }

    private static void downloadFileFromUrl(HttpClient client, String url,
                                            String localFilename, Path targetDir,
                                            boolean force, Consumer<String> logger,
                                            boolean required, int fileIndex,
                                            int totalFiles, ProgressListener progressListener)
            throws IOException, InterruptedException {

        Path target = targetDir.resolve(localFilename);

        if (Files.exists(target) && !force) {
            logger.accept("  Skipping (exists): " + localFilename);
            progressListener.onProgress(new ProgressEvent(localFilename, fileIndex, totalFiles,
                    Files.size(target), Files.size(target), true, true));
            return;
        }

        String sanitizedUrl = sanitizeUrl(url, localFilename, required, logger);
        if (sanitizedUrl == null) {
            progressListener.onProgress(new ProgressEvent(localFilename, fileIndex, totalFiles,
                    0L, 0L, true, true));
            return;
        }

        logger.accept("  Downloading: " + localFilename + " ...");

        HttpRequest request = HttpRequest.newBuilder()
                .uri(parseUriOrThrow(sanitizedUrl, localFilename, required))
                .timeout(Duration.ofMinutes(30))
                .GET()
                .build();

        HttpResponse<InputStream> response = client.send(request,
                HttpResponse.BodyHandlers.ofInputStream());

        if (response.statusCode() == 200) {
            long totalBytes = response.headers().firstValueAsLong("Content-Length").orElse(-1L);
            progressListener.onProgress(new ProgressEvent(localFilename, fileIndex, totalFiles,
                    0L, totalBytes, false, false));
            copyBodyWithProgress(response.body(), target, localFilename, totalBytes,
                    fileIndex, totalFiles, progressListener);
            long sizeKb = Files.size(target) / 1024;
            logger.accept("  Downloaded: " + localFilename + " (" + sizeKb + " KB)");
        } else if (response.statusCode() == 404 && !required) {
            logger.accept("  Optional file not found (skipped): " + localFilename);
            progressListener.onProgress(new ProgressEvent(localFilename, fileIndex, totalFiles,
                    0L, 0L, true, true));
        } else {
            String msg = "HTTP " + response.statusCode() + " for " + localFilename
                    + " (url: " + sanitizedUrl + ")";
            if (required) {
                throw new IOException(msg);
            } else {
                logger.accept("  Warning: " + msg);
                progressListener.onProgress(new ProgressEvent(localFilename, fileIndex, totalFiles,
                        0L, 0L, true, true));
            }
        }
    }

    private static void copyBodyWithProgress(InputStream body, Path target,
                                             String localFilename, long totalBytes,
                                             int fileIndex, int totalFiles,
                                             ProgressListener progressListener)
            throws IOException {
        Path parent = target.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path partial = target.resolveSibling(target.getFileName() + ".part");
        long bytesRead = 0L;
        try (InputStream in = body;
             OutputStream out = Files.newOutputStream(partial,
                     StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            byte[] buffer = new byte[DOWNLOAD_BUFFER_SIZE];
            int read;
            while ((read = in.read(buffer)) >= 0) {
                if (read == 0) {
                    continue;
                }
                out.write(buffer, 0, read);
                bytesRead += read;
                progressListener.onProgress(new ProgressEvent(localFilename, fileIndex, totalFiles,
                        bytesRead, totalBytes, false, false));
            }
        } catch (IOException ex) {
            Files.deleteIfExists(partial);
            throw ex;
        }
        Files.move(partial, target, StandardCopyOption.REPLACE_EXISTING);
        long finalTotalBytes = totalBytes > 0L ? totalBytes : bytesRead;
        progressListener.onProgress(new ProgressEvent(localFilename, fileIndex, totalFiles,
                bytesRead, finalTotalBytes, true, false));
    }

    private static String sanitizeUrl(String url, String localFilename, boolean required,
                                      Consumer<String> logger) throws IOException {
        String sanitized = url == null ? "" : url.trim();
        if (!sanitized.isEmpty()) {
            return sanitized;
        }
        if (required) {
            throw new IOException("Invalid required download URL for " + localFilename
                    + ": value is blank");
        }
        logger.accept("  Optional file URL is blank (skipped): " + localFilename);
        return null;
    }

    private static URI parseUriOrThrow(String url, String localFilename, boolean required)
            throws IOException {
        try {
            return URI.create(url);
        } catch (IllegalArgumentException ex) {
            throw new IOException("Invalid " + (required ? "required" : "optional")
                    + " download URL for " + localFilename + ": " + url, ex);
        }
    }
}
