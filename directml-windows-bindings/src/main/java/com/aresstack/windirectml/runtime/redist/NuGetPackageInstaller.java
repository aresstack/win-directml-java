package com.aresstack.windirectml.runtime.redist;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Generische Façade für „NuGet-Paket nach App-Settings-Ordner installieren":
 * lädt das Paket (falls noch nicht im Cache) und entpackt die gewünschten
 * Dateien.
 * <p>
 * Beide abhängigen Komponenten ({@link NuGetPackageDownloader},
 * {@link PackageExtractor}) sind injizierbar – Standardkombination ist
 * {@link NuGetV2PackageDownloader} + {@link ZipPackageExtractor}.
 */
public final class NuGetPackageInstaller {

    private static final Logger log = LoggerFactory.getLogger(NuGetPackageInstaller.class);

    private final NuGetPackageDownloader downloader;
    private final PackageExtractor extractor;

    public NuGetPackageInstaller() {
        this(new NuGetV2PackageDownloader(), new ZipPackageExtractor());
    }

    public NuGetPackageInstaller(NuGetPackageDownloader downloader, PackageExtractor extractor) {
        this.downloader = downloader;
        this.extractor = extractor;
    }

    /**
     * @return die absoluten Pfade aller extrahierten Dateien (in Reihenfolge
     *         der {@link NuGetInstallConfig#extractRules()})
     */
    public List<Path> install(NuGetInstallConfig config) throws IOException {
        Files.createDirectories(config.appSettingsFolder());
        log.info("Installing NuGet package {} {} into {}",
                config.packageSpec().packageId(),
                config.packageSpec().version(),
                config.appSettingsFolder());
        Path nupkg = downloader.download(config);
        return extractor.extract(nupkg, config);
    }
}

