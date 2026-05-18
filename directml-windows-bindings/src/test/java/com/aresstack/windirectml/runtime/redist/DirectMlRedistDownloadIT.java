package com.aresstack.windirectml.runtime.redist;

import com.aresstack.windirectml.windows.DirectMlBindings;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Opt-in Integrationstest: lädt Microsoft.AI.DirectML wirklich von
 * {@code nuget.org} und entpackt die x64-DLL. Wird nur ausgeführt, wenn
 * {@code -Dwindirectml.redist.itest=true} gesetzt ist.
 *
 * <pre>
 *   ./gradlew.bat :directml-windows-bindings:test \
 *       --tests *DirectMlRedistDownloadIT \
 *       -Dwindirectml.redist.itest=true
 * </pre>
 */
class DirectMlRedistDownloadIT {

    @Test
    @EnabledIfSystemProperty(named = "windirectml.redist.itest", matches = "true")
    void realDownloadFromNuGetOrg(@TempDir Path tmp) throws IOException {
        DirectMlRedistInstaller installer = new DirectMlRedistInstaller(tmp);
        Path dll = installer.install(DirectMlRedistInstaller.DEFAULT_VERSION, "x64");

        assertTrue(Files.isRegularFile(dll), "DirectML.dll must exist after real download");
        long size = Files.size(dll);
        assertTrue(size > 100_000L, "DirectML.dll suspiciously small (" + size + " bytes)");

        // Also verify cache layout.
        Path nupkg = tmp.resolve("nuget-cache/Microsoft.AI.DirectML/"
                + DirectMlRedistInstaller.DEFAULT_VERSION
                + "/Microsoft.AI.DirectML." + DirectMlRedistInstaller.DEFAULT_VERSION + ".nupkg");
        assertTrue(Files.isRegularFile(nupkg), "nupkg must remain in the cache for reuse");

        // Sanity: SYS_PROP must NOT have been touched by install() alone.
        // installAndUse() is the explicit opt-in for that side-effect.
        String prop = System.getProperty(DirectMlBindings.SYS_PROP_DIRECTML_DLL);
        // (No assertion needed here – just documenting intent.)
        assert prop == null || !prop.equals(dll.toString());
    }
}

