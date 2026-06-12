package com.aresstack.windirectml.inference.artifact;

import java.io.IOException;
import java.nio.file.Path;

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
            case PACKAGE_LEGACY_DIRECT -> ModelConversionAction.INSPECT;
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

    /** The standard "use the Download tab -> Convert" guidance for a not-ready status. */
    default String actionableMessage(ModelArtifactStatus status) {
        String base = "Missing or unusable " + family().displayName() + " runtime package ("
                + status.packageState() + (status.reason().isBlank() ? "" : ": " + status.reason()) + ").";
        if (!hasCompiler()) {
            return base + " This family runs directly from its raw source; "
                    + "download the model first from the Download tab.";
        }
        return base + " Use the Download tab -> Convert to build it; inference never compiles a package.";
    }
}
