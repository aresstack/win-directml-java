package com.aresstack.windirectml.inference.smollm2;

import java.util.List;
import java.util.Objects;

/**
 * Describes the immutable tensor upload surface for a prepared SmolLM2 WARP plan.
 *
 * <p>The manifest is Java-only metadata. It does not allocate native resources and can be used by a future native
 * executor to create upload/default buffers deterministically.</p>
 */
public record SmolLM2WarpUploadManifest(List<SmolLM2WarpBufferEntry> weightEntries,
                                        long totalUploadBytes,
                                        int alignmentBytes,
                                        List<String> warnings) {

    public SmolLM2WarpUploadManifest {
        weightEntries = List.copyOf(Objects.requireNonNull(weightEntries, "weightEntries"));
        if (totalUploadBytes < 0L) {
            throw new IllegalArgumentException("totalUploadBytes must not be negative");
        }
        if (alignmentBytes <= 0) {
            throw new IllegalArgumentException("alignmentBytes must be positive");
        }
        warnings = warnings == null ? List.of() : List.copyOf(warnings);
    }

    /**
     * Create the upload manifest from a prepared runtime plan.
     */
    public static SmolLM2WarpUploadManifest fromPlan(SmolLM2WarpRuntimePlan plan) {
        Objects.requireNonNull(plan, "plan");
        SmolLM2WarpBufferPlan bufferPlan = plan.bufferPlan();
        return new SmolLM2WarpUploadManifest(
                bufferPlan.entriesOfKind(SmolLM2WarpBufferKind.WEIGHT),
                bufferPlan.totalWeightBytes(),
                bufferPlan.alignmentBytes(),
                plan.warnings());
    }

    public int tensorCount() {
        return weightEntries.size();
    }

    public boolean readyForUpload() {
        return warnings.isEmpty() && !weightEntries.isEmpty();
    }
}
