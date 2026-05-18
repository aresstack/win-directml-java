package com.aresstack.windirectml.runtime.redist;

import java.util.Objects;

/**
 * Eine Datei aus einem NuGet-Paket, die an einen bestimmten relativen Pfad
 * unterhalb des App-Settings-Verzeichnisses entpackt werden soll.
 * <p>
 * Beispiel für die DirectML-Redistributable auf x64:
 * <pre>{@code
 * new ExtractRule(
 *     "bin/x64-win/DirectML.dll",
 *     "native/directml/1.15.4/x64-win/DirectML.dll");
 * }</pre>
 * <p>
 * Pfadangaben werden vor dem Vergleich auf {@code '/'} normalisiert; ein
 * {@link ZipPackageExtractor} darf nicht ausserhalb des App-Settings-
 * Verzeichnisses schreiben (Schutz vor ZIP-Slip).
 */
public record ExtractRule(String packageEntryPath, String targetRelativePath) {

    public ExtractRule {
        packageEntryPath = normalize(packageEntryPath, "packageEntryPath");
        targetRelativePath = requireText(targetRelativePath, "targetRelativePath")
                .replace('\\', '/');
        if (targetRelativePath.startsWith("/") || targetRelativePath.contains("..")) {
            throw new IllegalArgumentException(
                    "targetRelativePath must be a relative, non-traversing path: " + targetRelativePath);
        }
    }

    /** @return {@code true} when {@code actualEntryPath} matches this rule (after normalisation). */
    public boolean matches(String actualEntryPath) {
        return packageEntryPath.equalsIgnoreCase(normalize(actualEntryPath, "actualEntryPath"));
    }

    private static String normalize(String entryPath, String name) {
        return requireText(entryPath, name).replace('\\', '/');
    }

    private static String requireText(String value, String name) {
        Objects.requireNonNull(value, name);
        String trimmed = value.trim();
        if (trimmed.isEmpty()) throw new IllegalArgumentException(name + " must not be blank");
        return trimmed;
    }
}

