package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.RuntimeLoadability;

import java.util.Objects;

/**
 * Describes whether a SmolLM2 package can be executed by the available runtime path.
 */
public record SmolLM2RuntimeLoadability(
        boolean runtimeLoadable,
        String runtimeLoadMode,
        String runtimeLoadableReason
) {
    public SmolLM2RuntimeLoadability {
        runtimeLoadMode = requireText(runtimeLoadMode, "runtimeLoadMode");
        runtimeLoadableReason = requireText(runtimeLoadableReason, "runtimeLoadableReason");
    }

    public static SmolLM2RuntimeLoadability forPackage(SmolLM2LayoutReport layoutReport, boolean payloadIncluded) {
        Objects.requireNonNull(layoutReport, "layoutReport");
        if (!layoutReport.layoutComplete()) {
            return notLoadable("invalid-layout", SmolLM2LayoutReport.INCOMPLETE_LAYOUT);
        }
        if (!payloadIncluded) {
            return notLoadable("manifest-only", SmolLM2LayoutReport.MISSING_DENSE_PAYLOAD);
        }
        return loadable("reference", SmolLM2LayoutReport.REFERENCE_RUNTIME_AVAILABLE);
    }

    public static SmolLM2RuntimeLoadability loadable(String mode, String reason) {
        return new SmolLM2RuntimeLoadability(true, mode, reason);
    }

    public static SmolLM2RuntimeLoadability notLoadable(String mode, String reason) {
        return new SmolLM2RuntimeLoadability(false, mode, reason);
    }

    /**
     * Map onto the model-family-neutral {@link RuntimeLoadability} report (field-identical). Additive view; SmolLM2's
     * own behaviour is unchanged.
     */
    public RuntimeLoadability toRuntimeLoadability() {
        return new RuntimeLoadability(runtimeLoadable, runtimeLoadMode, runtimeLoadableReason);
    }

    private static String requireText(String value, String name) {
        Objects.requireNonNull(value, name);
        if (value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value;
    }
}
