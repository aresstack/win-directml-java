package com.aresstack.windirectml.inference.model;

import java.util.Objects;

/**
 * Model-family-neutral runtime-loadability report for a {@code .wdmlpack} package.
 *
 * <p>Whether a compiled package can be executed by the available runtime, plus the load mode and a human-readable
 * reason. Shared vocabulary so callers can interpret loadability the same way for the seq2seq (T5) and decoder-only
 * families. The fields mirror the existing per-family reports (e.g. {@code SmolLM2RuntimeLoadability}) so a family can
 * map onto this neutral type without renaming or changing its own.</p>
 */
public record RuntimeLoadability(boolean runtimeLoadable, String runtimeLoadMode, String runtimeLoadableReason) {

    public RuntimeLoadability {
        runtimeLoadMode = requireText(runtimeLoadMode, "runtimeLoadMode");
        runtimeLoadableReason = requireText(runtimeLoadableReason, "runtimeLoadableReason");
    }

    public static RuntimeLoadability loadable(String mode, String reason) {
        return new RuntimeLoadability(true, mode, reason);
    }

    public static RuntimeLoadability notLoadable(String mode, String reason) {
        return new RuntimeLoadability(false, mode, reason);
    }

    private static String requireText(String value, String name) {
        Objects.requireNonNull(value, name);
        if (value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value;
    }
}
