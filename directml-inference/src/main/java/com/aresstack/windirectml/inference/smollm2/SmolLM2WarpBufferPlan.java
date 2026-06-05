package com.aresstack.windirectml.inference.smollm2;

import java.util.List;
import java.util.Objects;

/**
 * Complete buffer layout prepared before any native WARP resource is allocated.
 */
public record SmolLM2WarpBufferPlan(List<SmolLM2WarpBufferEntry> entries,
                                    long totalWeightBytes,
                                    long totalKvCacheBytes,
                                    long totalScratchBytes,
                                    int alignmentBytes) {

    public SmolLM2WarpBufferPlan {
        entries = List.copyOf(Objects.requireNonNull(entries, "entries"));
        if (totalWeightBytes < 0L || totalKvCacheBytes < 0L || totalScratchBytes < 0L) {
            throw new IllegalArgumentException("buffer byte totals must not be negative");
        }
        if (alignmentBytes <= 0) {
            throw new IllegalArgumentException("alignmentBytes must be positive");
        }
    }

    public long totalBytes() {
        return totalWeightBytes + totalKvCacheBytes + totalScratchBytes;
    }

    public List<SmolLM2WarpBufferEntry> entriesOfKind(SmolLM2WarpBufferKind kind) {
        return entries.stream()
                .filter(entry -> entry.kind() == kind)
                .toList();
    }
}
