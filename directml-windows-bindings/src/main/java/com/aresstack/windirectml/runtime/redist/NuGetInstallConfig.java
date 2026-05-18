package com.aresstack.windirectml.runtime.redist;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Vollständige Installations-Konfiguration: woraus (NuGet-Paket), wohin
 * (App-Settings-Ordner) und welche Dateien (Extract-Regeln) entpackt werden
 * sollen.
 * <p>
 * Der App-Settings-Ordner wird beim Konstruieren absolut normalisiert; die
 * Regeln werden in einer unveränderlichen Liste gehalten.
 */
public record NuGetInstallConfig(Path appSettingsFolder,
                                  NuGetPackageSpec packageSpec,
                                  List<ExtractRule> extractRules) {

    public NuGetInstallConfig {
        Objects.requireNonNull(appSettingsFolder, "appSettingsFolder");
        Objects.requireNonNull(packageSpec, "packageSpec");
        Objects.requireNonNull(extractRules, "extractRules");
        if (extractRules.isEmpty()) {
            throw new IllegalArgumentException("extractRules must not be empty");
        }
        appSettingsFolder = appSettingsFolder.toAbsolutePath().normalize();
        extractRules = List.copyOf(extractRules);
    }

    /** Cache-Verzeichnis für heruntergeladene {@code .nupkg}-Dateien. */
    public Path nugetCacheFolder() {
        return appSettingsFolder.resolve("nuget-cache")
                .resolve(packageSpec.packageId())
                .resolve(packageSpec.version());
    }

    /** Volle Cache-Pfad für die {@code .nupkg}-Datei dieses Pakets. */
    public Path nupkgCacheFile() {
        return nugetCacheFolder().resolve(packageSpec.nupkgFileName());
    }
}

