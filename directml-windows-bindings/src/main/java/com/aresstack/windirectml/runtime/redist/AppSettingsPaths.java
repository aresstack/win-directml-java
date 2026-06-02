package com.aresstack.windirectml.runtime.redist;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;

/**
 * Bestimmt den Standard-App-Settings-Ordner pro Betriebssystem.
 * Verwendet wird auf Windows {@code %LOCALAPPDATA%\<appName>} (Geräte-lokal,
 * roamt nicht), unter Linux {@code $XDG_DATA_HOME/<appName>} bzw.
 * {@code ~/.local/share/<appName>}, unter macOS
 * {@code ~/Library/Application Support/<appName>}.
 * <p>
 * Diese Klasse legt <em>nichts</em> automatisch an – sie liefert nur
 * den absoluten Pfad. Anlegen erfolgt erst beim ersten Schreiben durch
 * den Installer.
 */
public final class AppSettingsPaths {

    private AppSettingsPaths() {
    }

    /**
     * @param appName logischer App-Name, wird als Verzeichnisname benutzt
     *                (z.&nbsp;B. {@code "win-directml-java"})
     */
    public static Path forApp(String appName) {
        if (appName == null || appName.isBlank()) {
            throw new IllegalArgumentException("appName must not be blank");
        }
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (os.contains("windows")) {
            String localAppData = System.getenv("LOCALAPPDATA");
            if (localAppData == null || localAppData.isBlank()) {
                localAppData = System.getProperty("user.home") + "\\AppData\\Local";
            }
            return Paths.get(localAppData, appName).toAbsolutePath().normalize();
        }
        if (os.contains("mac")) {
            return Paths.get(System.getProperty("user.home"),
                    "Library", "Application Support", appName).toAbsolutePath().normalize();
        }
        // Linux / *nix
        String xdg = System.getenv("XDG_DATA_HOME");
        if (xdg != null && !xdg.isBlank()) {
            return Paths.get(xdg, appName).toAbsolutePath().normalize();
        }
        return Paths.get(System.getProperty("user.home"), ".local", "share", appName)
                .toAbsolutePath().normalize();
    }

    /**
     * Default-Ordner für dieses Projekt: {@code "win-directml-java"}.
     */
    public static Path defaultForWinDirectMlJava() {
        return forApp("win-directml-java");
    }
}

