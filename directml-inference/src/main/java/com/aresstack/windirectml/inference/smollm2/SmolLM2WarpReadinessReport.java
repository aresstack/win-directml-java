package com.aresstack.windirectml.inference.smollm2;

import java.util.List;
import java.util.Objects;

/**
 * Read-only diagnostics for the prepared SmolLM2 WARP execution boundary.
 *
 * <p>The report intentionally does not imply that native execution is available. It only describes the prepared
 * resource and kernel surface that a configured {@link SmolLM2WarpExecutor} would receive.</p>
 */
public record SmolLM2WarpReadinessReport(boolean executable,
                                         String runtimeMode,
                                         String reason,
                                         int weightTensorCount,
                                         long totalUploadBytes,
                                         long totalKvCacheBytes,
                                         long totalScratchBytes,
                                         int alignmentBytes,
                                         int kernelStepCount,
                                         List<String> warnings) {

    public SmolLM2WarpReadinessReport {
        runtimeMode = runtimeMode == null || runtimeMode.isBlank() ? "warp" : runtimeMode;
        reason = reason == null ? "" : reason;
        if (weightTensorCount < 0) {
            throw new IllegalArgumentException("weightTensorCount must not be negative");
        }
        if (totalUploadBytes < 0L || totalKvCacheBytes < 0L || totalScratchBytes < 0L) {
            throw new IllegalArgumentException("byte totals must not be negative");
        }
        if (alignmentBytes <= 0) {
            throw new IllegalArgumentException("alignmentBytes must be positive");
        }
        if (kernelStepCount < 0) {
            throw new IllegalArgumentException("kernelStepCount must not be negative");
        }
        warnings = warnings == null ? List.of() : List.copyOf(warnings);
    }

    /**
     * Build a diagnostic report from a prepared WARP runtime.
     */
    public static SmolLM2WarpReadinessReport fromRuntime(SmolLM2WarpRuntime runtime) {
        Objects.requireNonNull(runtime, "runtime");
        SmolLM2WarpRuntimePlan plan = runtime.plan();
        SmolLM2WarpExecutionStatus status = runtime.status();
        SmolLM2WarpUploadManifest uploadManifest = runtime.uploadManifest();
        SmolLM2WarpBufferPlan bufferPlan = plan.bufferPlan();
        return new SmolLM2WarpReadinessReport(
                status.executable(),
                status.runtimeMode(),
                status.reason(),
                uploadManifest.tensorCount(),
                uploadManifest.totalUploadBytes(),
                bufferPlan.totalKvCacheBytes(),
                bufferPlan.totalScratchBytes(),
                uploadManifest.alignmentBytes(),
                plan.kernelPlan().steps().size(),
                status.warnings());
    }

    /**
     * Return true when the report describes a prepared but currently non-executable WARP path.
     */
    public boolean preparedButNotExecutable() {
        return !executable && weightTensorCount > 0 && totalUploadBytes > 0L && kernelStepCount > 0;
    }

    /**
     * Return true when DirectML/WARP probing reached a real adapter/context but kernels are still missing.
     */
    public boolean directMlProbeAvailable() {
        return reason.contains("DirectML/WARP probe succeeded");
    }

    /**
     * Return a concise human-readable byte amount.
     */
    public static String formatBytes(long bytes) {
        if (bytes < 1024L) {
            return bytes + " B";
        }
        double kib = bytes / 1024.0d;
        if (kib < 1024.0d) {
            return String.format(java.util.Locale.ROOT, "%.1f KiB", kib);
        }
        double mib = kib / 1024.0d;
        if (mib < 1024.0d) {
            return String.format(java.util.Locale.ROOT, "%.1f MiB", mib);
        }
        double gib = mib / 1024.0d;
        return String.format(java.util.Locale.ROOT, "%.2f GiB", gib);
    }
}
