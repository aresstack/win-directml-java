package com.aresstack.windirectml.runtime.redist;

import com.aresstack.windirectml.windows.DirectMlBindings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/**
 * Komfort-Façade: lädt das Microsoft.AI.DirectML-NuGet-Paket bei Bedarf in
 * den App-Settings-Ordner herunter und liefert den Pfad zu der entpackten
 * {@code DirectML.dll} zurück. Optional setzt der Aufrufer mit dem
 * gelieferten Pfad anschließend {@link DirectMlBindings#SYS_PROP_DIRECTML_DLL}
 * (oder benutzt {@link #installAndUse(String, String)} dafür).
 * <p>
 * <b>Policy (siehe {@link DirectMlBindings#SYS_PROP_DIRECTML_DLL}):</b>
 * Dieser Installer darf nur aufgerufen werden, wenn der <em>Endbenutzer</em>
 * dem Bezug eines Redistributables explizit zugestimmt hat (z.&nbsp;B. via
 * UI-Opt-In oder Konfigurationsschalter). Es wird niemals automatisch oder
 * still bei einem normalen Programmstart heruntergeladen, und niemals in
 * {@code C:\Windows\System32} oder einen anderen System-Ordner geschrieben –
 * nur in den App-eigenen lokalen Settings-Ordner.
 * <p>
 * Auflösungs-Layout im App-Settings-Ordner:
 * <pre>
 * &lt;AppSettings&gt;/
 *   nuget-cache/Microsoft.AI.DirectML/&lt;version&gt;/Microsoft.AI.DirectML.&lt;version&gt;.nupkg
 *   native/directml/&lt;version&gt;/&lt;arch&gt;-win/DirectML.dll
 * </pre>
 * Unterstützte Architekturen: {@code x64}, {@code x86}, {@code arm64}, {@code arm}.
 */
public final class DirectMlRedistInstaller {

    private static final Logger log = LoggerFactory.getLogger(DirectMlRedistInstaller.class);

    /** Default-Version, gegen die diese Codebase getestet wurde. */
    public static final String DEFAULT_VERSION = "1.15.4";

    /** NuGet-Paket-Id der Microsoft DirectML-Redistributable. */
    public static final String PACKAGE_ID = "Microsoft.AI.DirectML";

    private final NuGetPackageInstaller installer;
    private final Path appSettingsFolder;

    public DirectMlRedistInstaller() {
        this(AppSettingsPaths.defaultForWinDirectMlJava(), new NuGetPackageInstaller());
    }

    public DirectMlRedistInstaller(Path appSettingsFolder) {
        this(appSettingsFolder, new NuGetPackageInstaller());
    }

    public DirectMlRedistInstaller(Path appSettingsFolder, NuGetPackageInstaller installer) {
        this.appSettingsFolder = appSettingsFolder.toAbsolutePath().normalize();
        this.installer = installer;
    }

    /**
     * Baut eine NuGet-Install-Konfiguration für DirectML.
     */
    public static NuGetInstallConfig configFor(Path appSettingsFolder, String version, String arch) {
        String normalizedArch = normaliseArch(arch);
        ExtractRule rule = new ExtractRule(
                "bin/" + normalizedArch + "-win/DirectML.dll",
                "native/directml/" + version + "/" + normalizedArch + "-win/DirectML.dll");
        return new NuGetInstallConfig(appSettingsFolder,
                new NuGetPackageSpec(PACKAGE_ID, version),
                List.of(rule));
    }

    /**
     * Installiert die DirectML-Redist und liefert den Pfad zur {@code DirectML.dll}.
     * Idempotent: bestehende Cache- und Extract-Ergebnisse werden wiederverwendet.
     *
     * @param version NuGet-Version, z.&nbsp;B. {@code "1.15.4"}
     * @param arch    Architektur: {@code "x64"}, {@code "x86"}, {@code "arm64"} oder {@code "arm"}.
     *                Auch {@code "x64-win"} u.&nbsp;ä. werden akzeptiert.
     */
    public Path install(String version, String arch) throws IOException {
        NuGetInstallConfig config = configFor(appSettingsFolder, version, arch);
        List<Path> extracted = installer.install(config);
        Path dll = extracted.get(0);
        if (!Files.isRegularFile(dll)) {
            throw new IOException("DirectML.dll missing after installation: " + dll);
        }
        log.info("DirectML redist ready at {}", dll);
        return dll;
    }

    /**
     * Wie {@link #install(String, String)}, setzt aber zusätzlich
     * {@link DirectMlBindings#SYS_PROP_DIRECTML_DLL} auf den entpackten
     * Pfad. Muss <em>vor</em> der erstmaligen Verwendung von
     * {@code DirectMlBindings} aufgerufen werden, sonst greift der Symbol-
     * Lookup noch auf die in-box DLL zu (er ist nach dem ersten Zugriff
     * gecached).
     */
    public Path installAndUse(String version, String arch) throws IOException {
        Path dll = install(version, arch);
        System.setProperty(DirectMlBindings.SYS_PROP_DIRECTML_DLL, dll.toString());
        log.info("Set -D{}={}", DirectMlBindings.SYS_PROP_DIRECTML_DLL, dll);
        return dll;
    }

    public Path appSettingsFolder() {
        return appSettingsFolder;
    }

    private static String normaliseArch(String arch) {
        if (arch == null || arch.isBlank()) {
            throw new IllegalArgumentException("arch must not be blank");
        }
        String lower = arch.toLowerCase(Locale.ROOT).trim();
        if (lower.endsWith("-win")) lower = lower.substring(0, lower.length() - 4);
        return switch (lower) {
            case "x64", "amd64", "x86_64" -> "x64";
            case "x86", "i386", "i686"    -> "x86";
            case "arm64", "aarch64"        -> "arm64";
            case "arm"                     -> "arm";
            default -> throw new IllegalArgumentException(
                    "Unsupported arch '" + arch + "' (expected x64, x86, arm64 or arm)");
        };
    }
}

