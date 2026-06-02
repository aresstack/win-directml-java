package com.aresstack.windirectml.runtime.redist;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifiziert die NuGet-Installer-Pipeline ohne Netzzugriff: ein synthetisches
 * .nupkg (= ZIP) wird im Cache abgelegt, ein Stub-Downloader liefert es,
 * der Extractor entpackt die DirectML-DLL-Stand-in und der High-Level
 * {@link DirectMlRedistInstaller} liefert den korrekten Zielpfad.
 */
class DirectMlRedistInstallerTest {

    @Test
    void installExtractsDirectMlDllFromSyntheticNupkg(@TempDir Path tmp) throws IOException {
        byte[] dllPayload = "FAKE-DIRECTML-DLL-CONTENT".getBytes(StandardCharsets.UTF_8);

        // Stub downloader: writes a synthetic .nupkg into the expected cache path.
        NuGetPackageDownloader stubDownloader = config -> {
            Path nupkg = config.nupkgCacheFile();
            Files.createDirectories(nupkg.getParent());
            try (OutputStream out = Files.newOutputStream(nupkg);
                 ZipOutputStream zip = new ZipOutputStream(out)) {
                zip.putNextEntry(new ZipEntry("bin/x64-win/DirectML.dll"));
                zip.write(dllPayload);
                zip.closeEntry();
                zip.putNextEntry(new ZipEntry("bin/arm64-win/DirectML.dll"));
                zip.write(new byte[]{0, 1, 2});
                zip.closeEntry();
                zip.putNextEntry(new ZipEntry("include/DirectML.h"));
                zip.write("// header".getBytes(StandardCharsets.UTF_8));
                zip.closeEntry();
            }
            return nupkg;
        };

        NuGetPackageInstaller installer = new NuGetPackageInstaller(stubDownloader, new ZipPackageExtractor());
        DirectMlRedistInstaller redist = new DirectMlRedistInstaller(tmp, installer);

        Path dll = redist.install("1.15.4", "x64");

        Path expectedDll = tmp.resolve("native/directml/1.15.4/x64-win/DirectML.dll").normalize();
        assertEquals(expectedDll, dll, "installer must return the expected extract path");
        assertTrue(Files.isRegularFile(dll), "DLL stand-in must exist after install");
        assertArrayEquals(dllPayload, Files.readAllBytes(dll), "DLL bytes must match the entry payload");

        // The .nupkg must be cached.
        Path cached = tmp.resolve("nuget-cache/Microsoft.AI.DirectML/1.15.4/Microsoft.AI.DirectML.1.15.4.nupkg");
        assertTrue(Files.isRegularFile(cached), "nupkg must be cached at deterministic path");

        // A second install must be idempotent (no exception, same path returned).
        Path again = redist.install("1.15.4", "x64");
        assertEquals(dll, again, "repeated install must return the same path");
    }

    @Test
    void installRejectsUnknownArchitecture(@TempDir Path tmp) {
        DirectMlRedistInstaller redist = new DirectMlRedistInstaller(tmp);
        assertThrows(IllegalArgumentException.class,
                () -> redist.install("1.15.4", "mips"));
    }

    @Test
    void installFailsWhenRequiredEntryIsMissing(@TempDir Path tmp) throws IOException {
        NuGetPackageDownloader stubDownloader = config -> {
            Path nupkg = config.nupkgCacheFile();
            Files.createDirectories(nupkg.getParent());
            try (OutputStream out = Files.newOutputStream(nupkg);
                 ZipOutputStream zip = new ZipOutputStream(out)) {
                // No DirectML.dll inside.
                zip.putNextEntry(new ZipEntry("include/DirectML.h"));
                zip.write("// header".getBytes(StandardCharsets.UTF_8));
                zip.closeEntry();
            }
            return nupkg;
        };
        NuGetPackageInstaller installer = new NuGetPackageInstaller(stubDownloader, new ZipPackageExtractor());
        DirectMlRedistInstaller redist = new DirectMlRedistInstaller(tmp, installer);

        IOException ex = assertThrows(IOException.class,
                () -> redist.install("1.15.4", "x64"));
        assertTrue(ex.getMessage().contains("not found"),
                "error must mention missing entry, was: " + ex.getMessage());
    }

    @Test
    void extractorRejectsPathTraversal(@TempDir Path tmp) {
        assertThrows(IllegalArgumentException.class,
                () -> new ExtractRule("bin/x64-win/DirectML.dll", "../../escape/DirectML.dll"));
    }

    @Test
    void configFactoryBuildsTheExpectedPathLayout(@TempDir Path tmp) {
        NuGetInstallConfig cfg = DirectMlRedistInstaller.configFor(tmp, "1.15.4", "x64");
        assertEquals("Microsoft.AI.DirectML", cfg.packageSpec().packageId());
        assertEquals("1.15.4", cfg.packageSpec().version());
        assertEquals(1, cfg.extractRules().size());
        ExtractRule rule = cfg.extractRules().get(0);
        assertEquals("bin/x64-win/DirectML.dll", rule.packageEntryPath());
        assertEquals("native/directml/1.15.4/x64-win/DirectML.dll", rule.targetRelativePath());
        // Cache path layout.
        Path expectedCache = tmp.resolve("nuget-cache/Microsoft.AI.DirectML/1.15.4/Microsoft.AI.DirectML.1.15.4.nupkg")
                .toAbsolutePath().normalize();
        assertEquals(expectedCache, cfg.nupkgCacheFile());
    }

    @Test
    void appSettingsPathsResolveBelowUserHomeOrLocalAppData() {
        Path p = AppSettingsPaths.forApp("win-directml-java");
        assertTrue(p.isAbsolute(), "must be absolute");
        assertTrue(p.toString().endsWith("win-directml-java"),
                "must end with the app name: " + p);
    }
}

