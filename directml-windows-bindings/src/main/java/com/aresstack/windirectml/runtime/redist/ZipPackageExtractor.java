package com.aresstack.windirectml.runtime.redist;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Entpackt nur die in {@link ExtractRule}s aufgeführten Einträge eines
 * {@code .nupkg} (= ZIP) und schreibt sie nach
 * {@link NuGetInstallConfig#appSettingsFolder()}.
 * <p>
 * Sicherheitsmerkmale:
 * <ul>
 *   <li>Jede Ziel-Datei wird vor dem Schreiben geprüft, dass sie nach
 *       {@link Path#normalize()} unterhalb des App-Settings-Ordners liegt
 *       (Schutz vor ZIP-Slip-Angriffen).</li>
 *   <li>Dateien werden zuerst in einen {@code .part}-Pfad geschrieben und
 *       erst nach erfolgreichem Kopieren an die Zielposition verschoben.</li>
 *   <li>Liegt die Zieldatei bereits in korrekter Größe vor, wird sie nicht
 *       erneut entpackt (Idempotenz).</li>
 * </ul>
 */
public final class ZipPackageExtractor implements PackageExtractor {

    private static final Logger log = LoggerFactory.getLogger(ZipPackageExtractor.class);

    @Override
    public List<Path> extract(Path packageFile, NuGetInstallConfig config) throws IOException {
        Path root = config.appSettingsFolder();

        // Resolve & validate target paths up-front (also catches duplicate rules).
        Map<String, Path> entryPathToTarget = new HashMap<>();
        List<Path> orderedTargets = new ArrayList<>();
        for (ExtractRule rule : config.extractRules()) {
            Path target = resolveSafeTarget(root, rule.targetRelativePath());
            entryPathToTarget.put(rule.packageEntryPath().toLowerCase(java.util.Locale.ROOT), target);
            orderedTargets.add(target);
        }

        Map<String, Boolean> entryFound = new HashMap<>();
        for (String key : entryPathToTarget.keySet()) entryFound.put(key, Boolean.FALSE);

        try (InputStream raw = Files.newInputStream(packageFile);
             BufferedInputStream buffered = new BufferedInputStream(raw);
             ZipInputStream zip = new ZipInputStream(buffered)) {
            ZipEntry entry;
            while ((entry = zip.getNextEntry()) != null) {
                if (entry.isDirectory()) continue;
                String key = entry.getName().replace('\\', '/').toLowerCase(java.util.Locale.ROOT);
                Path target = entryPathToTarget.get(key);
                if (target == null) continue;

                if (Files.exists(target) && Files.size(target) == entry.getSize() && entry.getSize() > 0) {
                    log.info("Extract cache hit: {} ({} bytes)", target, entry.getSize());
                } else {
                    extractEntry(zip, target);
                    log.info("Extracted {} -> {} ({} bytes)",
                            entry.getName(), target, Files.size(target));
                }
                entryFound.put(key, Boolean.TRUE);
            }
        }

        List<String> missing = new ArrayList<>();
        for (Map.Entry<String, Boolean> e : entryFound.entrySet()) {
            if (!e.getValue()) missing.add(e.getKey());
        }
        if (!missing.isEmpty()) {
            throw new IOException("Entries not found in NuGet package " + packageFile
                    + ": " + missing);
        }
        return orderedTargets;
    }

    private static Path resolveSafeTarget(Path root, String relative) throws IOException {
        Path resolved = root.resolve(relative).normalize();
        if (!resolved.startsWith(root)) {
            throw new IOException("Refusing to extract outside the app-settings folder: " + relative);
        }
        return resolved;
    }

    private static void extractEntry(InputStream zip, Path target) throws IOException {
        Files.createDirectories(target.getParent());
        Path tmp = target.resolveSibling(target.getFileName() + ".part");
        Files.copy(zip, tmp, StandardCopyOption.REPLACE_EXISTING);
        try {
            Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException atomicFailed) {
            Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }
}

