package com.aresstack.windirectml.runtime.redist;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Entpackt ausgewählte Dateien aus einem ZIP/NUPKG in den App-Settings-Ordner.
 */
public interface PackageExtractor {

    /**
     * @param packageFile path to the cached {@code .nupkg}
     * @param config      installation configuration with the extract rules
     * @return absolute paths of all extracted files (one per rule, in order)
     */
    List<Path> extract(Path packageFile, NuGetInstallConfig config) throws IOException;
}

