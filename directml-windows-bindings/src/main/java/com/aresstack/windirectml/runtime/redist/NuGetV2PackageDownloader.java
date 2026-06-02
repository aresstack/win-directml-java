package com.aresstack.windirectml.runtime.redist;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;

/**
 * Lädt NuGet-Pakete über die offizielle V2-Flat-API von {@code nuget.org}:
 * <pre>https://www.nuget.org/api/v2/package/{id}/{version}</pre>
 * <p>
 * Verwendet den Java-11+-{@link HttpClient} (folgt 30x-Redirects
 * automatisch), schreibt zuerst in eine {@code .download}-Datei und
 * verschiebt sie nach erfolgreichem Empfang atomar in den Cache, damit
 * abgebrochene Downloads den Cache nicht korrumpieren.
 */
public final class NuGetV2PackageDownloader implements NuGetPackageDownloader {

    private static final Logger log = LoggerFactory.getLogger(NuGetV2PackageDownloader.class);
    private static final String PACKAGE_URL_TEMPLATE = "https://www.nuget.org/api/v2/package/%s/%s";
    private static final Duration CONNECT_TIMEOUT = Duration.ofSeconds(30);
    private static final Duration REQUEST_TIMEOUT = Duration.ofMinutes(5);

    private final HttpClient httpClient;

    public NuGetV2PackageDownloader() {
        this(HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .connectTimeout(CONNECT_TIMEOUT)
                .build());
    }

    /**
     * Test seam: lets tests inject a stub {@link HttpClient}.
     */
    public NuGetV2PackageDownloader(HttpClient httpClient) {
        this.httpClient = httpClient;
    }

    @Override
    public Path download(NuGetInstallConfig config) throws IOException {
        Path cacheFile = config.nupkgCacheFile();
        if (Files.exists(cacheFile) && Files.size(cacheFile) > 0L) {
            log.info("NuGet cache hit: {}", cacheFile);
            return cacheFile;
        }

        Files.createDirectories(cacheFile.getParent());
        Path tmp = cacheFile.resolveSibling(cacheFile.getFileName() + ".download");

        URI uri = URI.create(String.format(PACKAGE_URL_TEMPLATE,
                urlSegment(config.packageSpec().packageId()),
                urlSegment(config.packageSpec().version())));
        log.info("Downloading NuGet package {} {} from {}",
                config.packageSpec().packageId(), config.packageSpec().version(), uri);

        HttpRequest req = HttpRequest.newBuilder(uri)
                .timeout(REQUEST_TIMEOUT)
                .header("User-Agent", "win-directml-java/NuGetV2PackageDownloader")
                .header("Accept", "application/zip, application/octet-stream")
                .GET()
                .build();

        HttpResponse<Path> resp;
        try {
            resp = httpClient.send(req,
                    HttpResponse.BodyHandlers.ofFile(tmp,
                            java.nio.file.StandardOpenOption.CREATE,
                            java.nio.file.StandardOpenOption.TRUNCATE_EXISTING,
                            java.nio.file.StandardOpenOption.WRITE));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while downloading " + uri, e);
        }

        if (resp.statusCode() / 100 != 2) {
            Files.deleteIfExists(tmp);
            throw new IOException("NuGet download failed: HTTP " + resp.statusCode() + " for " + uri);
        }
        if (Files.size(tmp) == 0L) {
            Files.deleteIfExists(tmp);
            throw new IOException("NuGet download returned empty body for " + uri);
        }

        try {
            Files.move(tmp, cacheFile, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException atomicFailed) {
            Files.move(tmp, cacheFile, StandardCopyOption.REPLACE_EXISTING);
        }
        log.info("Cached NuGet package at {} ({} bytes)", cacheFile, Files.size(cacheFile));
        return cacheFile;
    }

    private static String urlSegment(String value) {
        // NuGet package IDs and versions never contain reserved URI chars,
        // but we still escape spaces conservatively.
        return value.replace(" ", "%20");
    }
}

