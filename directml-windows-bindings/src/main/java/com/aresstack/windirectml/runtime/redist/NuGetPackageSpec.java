package com.aresstack.windirectml.runtime.redist;

import java.util.Objects;

/**
 * Identifiziert ein NuGet-Paket über {@code id + version}, z.&nbsp;B.
 * {@code Microsoft.AI.DirectML 1.15.4}.
 * <p>
 * Versionen müssen exakt der NuGet-Version entsprechen (semver-ähnlich,
 * inkl. optionaler Pre-Release-Suffixe). Wildcards werden nicht unterstützt.
 */
public record NuGetPackageSpec(String packageId, String version) {

    public NuGetPackageSpec {
        packageId = requireText(packageId, "packageId");
        version = requireText(version, "version");
    }

    private static String requireText(String value, String fieldName) {
        Objects.requireNonNull(value, fieldName);
        String trimmed = value.trim();
        if (trimmed.isEmpty()) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
        return trimmed;
    }

    /** Dateiname im NuGet-Cache: {@code <id>.<version>.nupkg}. */
    public String nupkgFileName() {
        return packageId + "." + version + ".nupkg";
    }
}

