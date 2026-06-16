package com.aresstack.windirectml.inference.gemma;

/**
 * Selects how {@link Gemma3WarpGenerator} drives the {@link Gemma3WarpDecodeSession} (GEMMA-WARP-13b-4):
 * the GPU-{@link #RESIDENT} / batched decode path (13b-3a/13b-3b — intermediates stay GPU-resident, fences
 * coalesced per layer) or the older {@link #SYNC} {@code float[]} {@code decodeStep} path (kept as a
 * debug/fallback).
 *
 * <p>The native-warp product path defaults to {@link #RESIDENT}; it can be overridden with
 * {@code -Dgemma.warp.execution=sync|resident}. Both paths are numerically identical (the resident path is
 * validated equal to the {@code float[]} oracle); resident does far fewer fence waits / readbacks per token.</p>
 */
public enum Gemma3WarpExecutionMode {

    /** Synchronous {@code float[]} decodeStep path (the original product path; debug/fallback). */
    SYNC("sync"),
    /** GPU-resident, per-layer batched decode path (13b-3a/13b-3b); the native-warp default. */
    RESIDENT("resident-batched");

    /** {@code -Dgemma.warp.execution=sync|resident}. */
    public static final String PROPERTY = "gemma.warp.execution";

    private final String displayLabel;

    Gemma3WarpExecutionMode(String displayLabel) {
        this.displayLabel = displayLabel;
    }

    public String displayLabel() {
        return displayLabel;
    }

    public boolean isResident() {
        return this == RESIDENT;
    }

    /** Resolve from {@code -Dgemma.warp.execution}; defaults to {@link #RESIDENT} when unset/unknown. */
    public static Gemma3WarpExecutionMode fromSystemProperty() {
        return fromValue(System.getProperty(PROPERTY));
    }

    /** Parse a mode value: only an explicit sync/synchronous/float/legacy/debug → {@link #SYNC}. */
    public static Gemma3WarpExecutionMode fromValue(String value) {
        if (value == null) {
            return RESIDENT;
        }
        switch (value.trim().toLowerCase()) {
            case "sync":
            case "synchronous":
            case "float":
            case "float[]":
            case "legacy":
            case "debug":
                return SYNC;
            default: // resident / batched / resident-batched / empty / unknown
                return RESIDENT;
        }
    }
}
