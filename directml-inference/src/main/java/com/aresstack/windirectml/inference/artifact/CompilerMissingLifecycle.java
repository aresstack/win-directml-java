package com.aresstack.windirectml.inference.artifact;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Lifecycle for families that have no {@code .wdmlpack} compiler/loader yet. They stay visible and
 * downloadable but are a homogeneous <b>not-executable</b> state: the package state is always
 * {@link PackageState#PACKAGE_COMPILER_MISSING}, conversion is unsupported, and inference always
 * fails fast. This is deliberately <em>not</em> a legacy path that runs directly from raw weights.
 */
public final class CompilerMissingLifecycle implements ModelPackageLifecycle {

    private final ModelFamily family;
    private final List<String> requiredFiles;
    private final List<List<String>> anyOfGroups;

    public CompilerMissingLifecycle(ModelFamily family, List<String> requiredFiles, List<List<String>> anyOfGroups) {
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
        // Even with valid raw files the model is not executable: there is no wdmlpack compiler/loader.
        return new ModelArtifactStatus(family, normalize(modelDir), raw.state(),
                PackageState.PACKAGE_COMPILER_MISSING, false, false,
                "package compiler not implemented (downloadable, not executable)");
    }

    @Override
    public ModelConversionResult convert(Path modelDir, boolean force) throws IOException {
        return ModelConversionResult.failed(
                "Package compiler not implemented for " + family.displayName()
                        + ". This model is downloadable but not executable until a wdmlpack compiler exists.");
    }

    private static Path normalize(Path modelDir) {
        return modelDir == null ? Path.of(".").toAbsolutePath().normalize() : modelDir.toAbsolutePath().normalize();
    }
}
