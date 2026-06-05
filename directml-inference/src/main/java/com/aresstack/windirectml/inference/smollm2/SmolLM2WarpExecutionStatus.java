package com.aresstack.windirectml.inference.smollm2;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Describes whether a prepared SmolLM2 WARP execution path can actually run.
 */
public record SmolLM2WarpExecutionStatus(boolean executable,
                                         String runtimeMode,
                                         String reason,
                                         List<String> warnings) {

    public static final String NATIVE_EXECUTOR_MISSING =
            "SmolLM2 WARP executor is not implemented yet; use reference runtime or AUTO fallback.";

    public SmolLM2WarpExecutionStatus {
        runtimeMode = runtimeMode == null ? "warp" : runtimeMode;
        reason = reason == null ? "" : reason;
        warnings = warnings == null ? List.of() : List.copyOf(warnings);
    }

    public static SmolLM2WarpExecutionStatus unsupported(SmolLM2WarpRuntimePlan plan) {
        List<String> mergedWarnings = new ArrayList<>();
        if (plan != null) {
            mergedWarnings.addAll(plan.warnings());
        }
        mergedWarnings.add(NATIVE_EXECUTOR_MISSING);
        return new SmolLM2WarpExecutionStatus(false, "warp", NATIVE_EXECUTOR_MISSING, mergedWarnings);
    }

    public static SmolLM2WarpExecutionStatus unsupported(String reason) {
        String safeReason = Objects.requireNonNull(reason, "reason");
        return new SmolLM2WarpExecutionStatus(false, "warp", safeReason, List.of(safeReason));
    }

    public boolean requiresFallback() {
        return !executable;
    }
}
