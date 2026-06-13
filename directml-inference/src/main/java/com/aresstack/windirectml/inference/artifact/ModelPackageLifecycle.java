package com.aresstack.windirectml.inference.artifact;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

/**
 * Per-family adapter over the existing compilers and runtime-package facades. Implementations
 * map a model directory to a {@link ModelArtifactStatus}, plan a conversion, and perform the
 * one-and-only package write.
 *
 * <p>Contract (enforced by tests):</p>
 * <ul>
 *   <li>{@link #inspect(Path)} and {@link #planConversion(Path)} must never write anything;</li>
 *   <li>{@link #convert(Path, boolean)} is the only method allowed to create a {@code .wdmlpack};</li>
 *   <li>{@link #validateOrThrowBeforeInference(Path)} must never compile.</li>
 * </ul>
 */
public interface ModelPackageLifecycle {

    ModelFamily family();

    /** Whether this family has a real {@code .wdmlpack} compiler (false = direct-source legacy). */
    boolean hasCompiler();

    /** Where {@link #convert(Path, boolean)} would write the package, or {@code null} for legacy families. */
    Path defaultPackagePath(Path modelDir);

    /**
     * The existing runtime package to load for inference, if one is present on disk. Never compiles.
     * Default resolves {@link #defaultPackagePath(Path)}; families that accept alternate file names
     * (e.g. T5) override this to search the directory.
     */
    default Optional<Path> existingPackage(Path modelDir) {
        Path candidate = defaultPackagePath(modelDir);
        return candidate != null && Files.isRegularFile(candidate) ? Optional.of(candidate) : Optional.empty();
    }

    /** Inspect raw + package state. Never writes, never compiles. */
    ModelArtifactStatus inspect(Path modelDir);

    /** Compile the raw source into a runtime package. The only write path. */
    ModelConversionResult convert(Path modelDir, boolean force) throws IOException;

    /** Short description of the raw source that would be compiled (for the conversion plan). */
    default String sourceDescription(Path modelDir) {
        return modelDir == null ? "" : modelDir.toString();
    }

    /** Derive the state-dependent conversion plan from {@link #inspect(Path)}. Never writes. */
    default ModelConversionPlan planConversion(Path modelDir) {
        if (!hasCompiler()) {
            return new ModelConversionPlan(ModelConversionAction.NOT_SUPPORTED, null,
                    sourceDescription(modelDir),
                    family().displayName() + " has no runtime-package compiler yet (direct-source legacy path)");
        }
        ModelArtifactStatus status = inspect(modelDir);
        Path target = defaultPackagePath(modelDir);
        ModelConversionAction action = chooseAction(status);
        return new ModelConversionPlan(action,
                action.writesPackage() ? target : null,
                sourceDescription(modelDir),
                status.reason());
    }

    private static ModelConversionAction chooseAction(ModelArtifactStatus status) {
        if (status.rawState() != RawAssetState.RAW_VALID && status.packageState() != PackageState.PACKAGE_VALID) {
            // Cannot (re)build without raw source and there is no usable package: only show status.
            return ModelConversionAction.INSPECT;
        }
        return switch (status.packageState()) {
            case PACKAGE_MISSING -> ModelConversionAction.CONVERT;
            case PACKAGE_STALE -> ModelConversionAction.RECONVERT;
            case PACKAGE_CORRUPT, PACKAGE_NOT_LOADABLE, PACKAGE_NOT_EXECUTABLE -> ModelConversionAction.REPAIR;
            case PACKAGE_VALID -> ModelConversionAction.RECONVERT;
            case PACKAGE_COMPILER_MISSING -> ModelConversionAction.NOT_SUPPORTED;
        };
    }

    /**
     * Throw an actionable error when inference must not run. Never compiles. The message points the
     * operator at the manual conversion flow rather than silently building a package.
     */
    default void validateOrThrowBeforeInference(Path modelDir) {
        ModelArtifactStatus status = inspect(modelDir);
        if (status.ready()) {
            return;
        }
        throw new IllegalStateException(actionableMessage(status));
    }

    /** Actionable, state-specific guidance for a not-ready status. Never implies a direct-from-source run. */
    default String actionableMessage(ModelArtifactStatus status) {
        String family = family().displayName();
        if (!hasCompiler() || status.packageState() == PackageState.PACKAGE_COMPILER_MISSING) {
            return "Package compiler not implemented for " + family
                    + ". This model is downloadable but not executable until a wdmlpack compiler exists.";
        }
        if (status.packageState() == PackageState.PACKAGE_MISSING) {
            return "Missing " + family + " runtime package. Use Download tab -> Check, then Convert.";
        }
        String reason = status.reason().isBlank() ? status.packageState().name() : status.reason();
        return "Invalid " + family + " runtime package (" + reason
                + "). Use Download tab -> Check, then Repair/Reconvert.";
    }
}
