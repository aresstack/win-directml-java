package com.aresstack.windirectml.workbench.artifact;

import com.aresstack.windirectml.inference.artifact.ModelArtifactStatus;
import com.aresstack.windirectml.inference.artifact.ModelConversionAction;
import com.aresstack.windirectml.inference.artifact.ModelConversionPlan;
import com.aresstack.windirectml.inference.artifact.ModelConversionResult;
import com.aresstack.windirectml.inference.artifact.ModelFamily;
import com.aresstack.windirectml.inference.artifact.ModelPackageLifecycle;
import com.aresstack.windirectml.inference.artifact.PackageState;
import com.aresstack.windirectml.inference.artifact.RawAssetState;

import java.nio.file.Path;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Swing-free controller behind one DownloadPanel model row. It maps a model row to a central
 * {@link ModelPackageLifecycle} (the W1 artifact API) and exposes exactly three operations:
 * {@link #inspect()} / {@link #plan()} (read-only, never write) and {@link #convert()} (the only
 * write path). {@link #refresh()} computes the row's visible status text and the state-dependent
 * Convert button view, so the UI holds no validation logic of its own.
 *
 * <p>The lifecycle is provided by a supplier so families whose package path depends on mutable UI
 * state (Qwen's selected ONNX variant) always resolve the current, variant-specific package.</p>
 */
public final class ModelArtifactRow {

    private final ModelFamily family;
    private final Supplier<Path> modelDir;
    private final Supplier<ModelPackageLifecycle> lifecycle;

    public ModelArtifactRow(ModelFamily family, Supplier<Path> modelDir, Supplier<ModelPackageLifecycle> lifecycle) {
        this.family = Objects.requireNonNull(family, "family");
        this.modelDir = Objects.requireNonNull(modelDir, "modelDir");
        this.lifecycle = Objects.requireNonNull(lifecycle, "lifecycle");
    }

    public ModelFamily family() {
        return family;
    }

    public Path modelDir() {
        return modelDir.get();
    }

    /** Read-only inspection. Never writes/compiles. */
    public ModelArtifactStatus inspect() {
        return lifecycle.get().inspect(modelDir.get());
    }

    /** Read-only conversion plan. Never writes/compiles. */
    public ModelConversionPlan plan() {
        return lifecycle.get().planConversion(modelDir.get());
    }

    /** The only write path: compile/repair the runtime package. */
    public ModelConversionResult convert() {
        ModelPackageLifecycle lc = lifecycle.get();
        Path dir = modelDir.get();
        if (!lc.hasCompiler()) {
            return ModelConversionResult.failed("Package compiler not implemented for "
                    + family.displayName() + ". This model is downloadable but not executable until a"
                    + " wdmlpack compiler exists.");
        }
        ModelConversionPlan p = lc.planConversion(dir);
        boolean force = p.action() == ModelConversionAction.RECONVERT
                || p.action() == ModelConversionAction.REPAIR;
        try {
            return lc.convert(dir, force);
        } catch (Exception e) {
            return ModelConversionResult.failed(family.displayName() + " conversion failed: " + e.getMessage());
        }
    }

    /** Compute the current status text + Convert button view (one inspect + one plan, no writes). */
    public RowView refresh() {
        ModelArtifactStatus status = inspect();
        ModelConversionPlan plan = plan();
        return new RowView(statusText(status), convertLabel(status, plan), convertEnabled(status, plan),
                convertTooltip(status, plan), status.ready(), status, plan);
    }

    private String statusText(ModelArtifactStatus s) {
        if (s.packageState() == PackageState.PACKAGE_COMPILER_MISSING) {
            return compilerMissingStatusText(s);
        }
        StringBuilder sb = new StringBuilder(family.displayName())
                .append(" — raw: ").append(pretty(s.rawState()))
                .append(", package: ").append(pretty(s.packageState()));
        if (s.ready()) {
            sb.append(" — READY");
        } else if (!s.reason().isBlank()) {
            sb.append(" — ").append(s.reason());
        }
        return sb.toString();
    }

    private String compilerMissingStatusText(ModelArtifactStatus s) {
        String reason = family == ModelFamily.GEMMA3 && !s.reason().isBlank()
                ? s.reason()
                : "package compiler not implemented (downloadable, not executable)";
        return new StringBuilder(family.displayName())
                .append(" — raw: ").append(pretty(s.rawState()))
                .append(" — ").append(reason)
                .toString();
    }

    private static String convertLabel(ModelArtifactStatus s, ModelConversionPlan p) {
        return switch (p.action()) {
            case CONVERT -> "Convert";
            case RECONVERT -> "Reconvert";
            case REPAIR -> "Repair package";
            case NOT_SUPPORTED -> downloadOnlyCandidate(p) ? "Download only" : "Compiler missing";
            case INSPECT -> rawUnavailable(s) ? "Download first" : "Inspect";
        };
    }

    private static boolean convertEnabled(ModelArtifactStatus s, ModelConversionPlan p) {
        return switch (p.action()) {
            case CONVERT, RECONVERT, REPAIR -> true;
            case NOT_SUPPORTED -> false;          // legacy direct: no real package compiler
            case INSPECT -> !rawUnavailable(s);    // nothing to convert until the raw source is present
        };
    }

    private static String convertTooltip(ModelArtifactStatus s, ModelConversionPlan p) {
        return switch (p.action()) {
            case CONVERT -> "Build the runtime package from the downloaded source";
            case RECONVERT -> "Rebuild the runtime package (" + p.reason() + ")";
            case REPAIR -> "Rebuild the unusable package (" + p.reason() + ")";
            case NOT_SUPPORTED -> downloadOnlyCandidate(p)
                    ? p.reason()
                    : "Package compiler not implemented for this family — downloadable, not executable";
            case INSPECT -> rawUnavailable(s)
                    ? "Download the raw model files first, then Convert"
                    : "Re-check the current artifact status";
        };
    }

    private static boolean downloadOnlyCandidate(ModelConversionPlan p) {
        String reason = p.reason().toLowerCase(java.util.Locale.ROOT);
        return reason.contains("download/probe") || reason.contains("download-only");
    }

    private static boolean rawUnavailable(ModelArtifactStatus s) {
        return s.rawState() != RawAssetState.RAW_VALID;
    }

    private static String pretty(RawAssetState state) {
        return state.name().toLowerCase().replace("raw_", "");
    }

    private static String pretty(PackageState state) {
        return state.name().toLowerCase().replace("package_", "");
    }

    /** Immutable view the UI renders without re-deriving any status logic. */
    public record RowView(String statusText,
                          String convertLabel,
                          boolean convertEnabled,
                          String convertTooltip,
                          boolean ready,
                          ModelArtifactStatus status,
                          ModelConversionPlan plan) {
    }
}
