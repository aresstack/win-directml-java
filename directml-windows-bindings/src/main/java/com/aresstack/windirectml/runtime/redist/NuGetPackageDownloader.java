package com.aresstack.windirectml.runtime.redist;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Lädt das {@code .nupkg} eines NuGet-Pakets in den Cache des
 * App-Settings-Ordners und liefert den lokalen Pfad zurück.
 * <p>
 * Implementierungen müssen idempotent sein: liegt das Paket bereits
 * vollständig vor, darf kein erneuter Download stattfinden.
 */
public interface NuGetPackageDownloader {

    /**
     * @param config installation configuration
     * @return absolute path to the cached {@code .nupkg}
     */
    Path download(NuGetInstallConfig config) throws IOException;
}

