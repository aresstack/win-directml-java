package com.aresstack.windirectml.windows;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Immutable GPU timestamp profile for a single submitted command list.
 */
public final class GpuTimestampProfile {

    private final long frequency;
    private final List<String> labels;
    private final long[] ticks;
    private final Map<String, Long> elapsedNanosByStage;

    GpuTimestampProfile(long frequency, List<String> labels, long[] ticks) {
        this.frequency = frequency;
        this.labels = Collections.unmodifiableList(new ArrayList<>(labels));
        this.ticks = ticks.clone();
        this.elapsedNanosByStage = Collections.unmodifiableMap(calculateStageDurations());
    }

    public boolean isEmpty() {
        return labels.size() < 2;
    }

    public long frequency() {
        return frequency;
    }

    public List<String> labels() {
        return labels;
    }

    public Map<String, Long> elapsedNanosByStage() {
        return elapsedNanosByStage;
    }

    public long elapsedNanos(String stage) {
        Long value = elapsedNanosByStage.get(stage);
        return value == null ? 0L : value;
    }

    public long totalNanos() {
        if (ticks.length < 2) {
            return 0L;
        }
        return ticksToNanos(Math.max(0L, ticks[ticks.length - 1] - ticks[0]));
    }

    public String formatSummary() {
        if (isEmpty()) {
            return "[Qwen2 GPU Timestamp Profile] no timestamp data";
        }

        StringBuilder sb = new StringBuilder(512);
        sb.append(String.format("[Qwen2 GPU Timestamp Profile] %.1f ms total, frequency=%d Hz%n",
                totalNanos() / 1e6, frequency));

        appendStageLine(sb, "input_upload");
        appendStageLine(sb, "input_norm");
        appendStageLine(sb, "qkv_proj");
        appendStageLine(sb, "qkv_bias");
        appendStageLine(sb, "rope_append");
        appendStageLine(sb, "attention");
        appendStageLine(sb, "o_proj");
        appendStageLine(sb, "residual1");
        appendStageLine(sb, "post_norm");
        appendStageLine(sb, "gate_up");
        appendStageLine(sb, "swiglu");
        appendStageLine(sb, "down_proj");
        appendStageLine(sb, "residual2");
        appendStageLine(sb, "final_norm");
        appendStageLine(sb, "readback_copy");
        return sb.toString();
    }

    private Map<String, Long> calculateStageDurations() {
        Map<String, Long> result = new LinkedHashMap<>();
        for (int index = 1; index < labels.size() && index < ticks.length; index++) {
            long deltaTicks = Math.max(0L, ticks[index] - ticks[index - 1]);
            long deltaNanos = ticksToNanos(deltaTicks);
            String stage = stageNameFor(labels.get(index));
            Long previous = result.get(stage);
            result.put(stage, previous == null ? deltaNanos : previous + deltaNanos);
        }
        return result;
    }

    private void appendStageLine(StringBuilder sb, String stage) {
        long nanos = elapsedNanos(stage);
        if (nanos == 0L) {
            return;
        }
        double percent = totalNanos() > 0 ? (100.0 * nanos / totalNanos()) : 0.0;
        sb.append(String.format("  %-13s %.1f ms (%.0f%%)%n", stage + ":", nanos / 1e6, percent));
    }

    private String stageNameFor(String markerLabel) {
        int separator = markerLabel.lastIndexOf('.');
        if (separator < 0 || separator == markerLabel.length() - 1) {
            return markerLabel;
        }
        return markerLabel.substring(separator + 1);
    }

    private long ticksToNanos(long deltaTicks) {
        if (frequency <= 0L) {
            return 0L;
        }
        return (long) ((deltaTicks * 1_000_000_000.0d) / frequency);
    }
}
