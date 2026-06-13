package com.aresstack.windirectml.workbench.artifact;

import com.aresstack.windirectml.inference.artifact.ModelArtifactStatus;
import com.aresstack.windirectml.inference.artifact.ModelFamily;
import com.aresstack.windirectml.inference.artifact.ModelPackageLifecycle;

import java.nio.file.Path;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Single source of truth for one visible Workbench model: where its raw files live, which package the
 * runtime loads, and the lifecycle that inspects/converts it. DownloadPanel and the runtime tabs both
 * resolve paths through this descriptor so the panel's status can never disagree with what the runtime
 * actually loads.
 *
 * <p>The package path is derived from the lifecycle ({@link ModelPackageLifecycle#defaultPackagePath})
 * - the exact path the runtime loader resolves - so "panel path == runtime path" holds by construction.
 * Suppliers are used because some values depend on mutable UI state (e.g. Qwen's selected variant).</p>
 */
public record ModelRuntimeDescriptor(ModelFamily family,
                                     String modelId,
                                     String displayName,
                                     Supplier<Path> rawDirectory,
                                     Supplier<ModelPackageLifecycle> lifecycle,
                                     RuntimeEntryPoint runtimeEntryPoint) {

    public ModelRuntimeDescriptor {
        family = Objects.requireNonNull(family, "family");
        Objects.requireNonNull(modelId, "modelId");
        Objects.requireNonNull(rawDirectory, "rawDirectory");
        Objects.requireNonNull(lifecycle, "lifecycle");
        runtimeEntryPoint = runtimeEntryPoint == null ? RuntimeEntryPoint.NONE : runtimeEntryPoint;
    }

    /** The raw/conversion source directory. */
    public Path rawDir() {
        return rawDirectory.get();
    }

    /** A fresh lifecycle (variant-correct). */
    public ModelPackageLifecycle newLifecycle() {
        return lifecycle.get();
    }

    /** The exact runtime package path the loader resolves, or {@code null} for compiler-missing families. */
    public Path runtimePackagePath() {
        return lifecycle.get().defaultPackagePath(rawDir());
    }

    /** A {@link ModelArtifactRow} bound to this descriptor for the DownloadPanel. */
    public ModelArtifactRow toRow() {
        return new ModelArtifactRow(family, rawDirectory, lifecycle);
    }

    /** Read-only status (never writes). */
    public ModelArtifactStatus inspect() {
        return lifecycle.get().inspect(rawDir());
    }

    /** Throw an actionable error if not executable. Never compiles. */
    public void validateOrThrowBeforeInference() {
        lifecycle.get().validateOrThrowBeforeInference(rawDir());
    }

    /** Where this descriptor's model is consumed at inference time. */
    public enum RuntimeEntryPoint {
        SUMMARIZER, EMBEDDINGS, RERANKER, NONE
    }
}
