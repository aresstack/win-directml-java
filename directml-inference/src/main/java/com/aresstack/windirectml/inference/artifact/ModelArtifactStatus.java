package com.aresstack.windirectml.inference.artifact;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Unified, device-free snapshot of a model's artifact state: the raw source state, the
 * runtime-package state, whether it is executable, and whether the family has a compiler.
 *
 * <p>This type is produced only by {@code inspect()} and never by a code path that writes
 * a package. {@link #ready()} is the single derived predicate that inference entry points
 * use ("can I run this now?").</p>
 */
public record ModelArtifactStatus(ModelFamily family,
                                  Path modelDir,
                                  RawAssetState rawState,
                                  PackageState packageState,
                                  boolean executable,
                                  boolean hasCompiler,
                                  String reason) {

    public ModelArtifactStatus {
        family = Objects.requireNonNull(family, "family");
        modelDir = Objects.requireNonNull(modelDir, "modelDir");
        rawState = Objects.requireNonNull(rawState, "rawState");
        packageState = Objects.requireNonNull(packageState, "packageState");
        reason = reason == null ? "" : reason;
    }

    /**
     * Whether inference can run now. Every visible model is treated homogeneously: inference requires a
     * valid, executable runtime package. Families without a package compiler
     * ({@link PackageState#PACKAGE_COMPILER_MISSING}) are never ready - they are visible but
     * not-executable, never a silent direct-from-source path.
     */
    public boolean ready() {
        return packageState == PackageState.PACKAGE_VALID && executable;
    }
}
