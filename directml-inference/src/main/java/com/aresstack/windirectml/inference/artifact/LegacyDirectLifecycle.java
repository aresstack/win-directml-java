package com.aresstack.windirectml.inference.artifact;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Lifecycle for families that have no {@code .wdmlpack} compiler yet and run directly from their
 * raw source (encoder/reranker today, Phi-3 via ONNX). The package state is always
 * {@link PackageState#PACKAGE_LEGACY_DIRECT} - an explicit, non-hidden marker - and conversion is
 * unsupported. Readiness depends only on the raw source being valid.
 */
public final class LegacyDirectLifecycle implements ModelPackageLifecycle {

    private final ModelFamily family;
    private final List<String> requiredFiles;
    private final List<List<String>> anyOfGroups;

    public LegacyDirectLifecycle(ModelFamily family, List<String> requiredFiles, List<List<String>> anyOfGroups) {
        this.family = Objects.requireNonNull(family, "family");
        this.requiredFiles = List.copyOf(requiredFiles);
        this.anyOfGroups = List.copyOf(anyOfGroups);
    }

    @Override
    public ModelFamily family() {
        return family;
    }

    @Override
    public boolean hasCompiler() {
        return false;
    }

    @Override
    public Path defaultPackagePath(Path modelDir) {
        return null;
    }

    @Override
    public ModelArtifactStatus inspect(Path modelDir) {
        RawAssetInspection.Result raw = RawAssetInspection.inspect(modelDir, requiredFiles, anyOfGroups);
        boolean ready = raw.state() == RawAssetState.RAW_VALID;
        String reason = ready
                ? "direct-source legacy path (no wdmlpack compiler yet); runs from raw files"
                : raw.detail();
        return new ModelArtifactStatus(family, normalize(modelDir), raw.state(),
                PackageState.PACKAGE_LEGACY_DIRECT, ready, false, reason);
    }

    @Override
    public ModelConversionResult convert(Path modelDir, boolean force) throws IOException {
        return ModelConversionResult.failed(
                "Package compiler not implemented for " + family.displayName()
                        + " (direct SafeTensors legacy path).");
    }

    private static Path normalize(Path modelDir) {
        return modelDir == null ? Path.of(".").toAbsolutePath().normalize() : modelDir.toAbsolutePath().normalize();
    }
}
