package com.aresstack.windirectml.inference.decoderonly;

/**
 * Family-neutral execution mode for decoder-only runtimes.
 *
 * <p>The shared concept behind each family's runtime-mode selection: run the CPU reference, the WARP projection path,
 * or auto-select WARP with a reference fallback. Families keep their own mode type for family-specific display labels
 * and map onto this one (the adapter point a second family such as Qwen reuses). The labels here are generic and
 * deliberately do not describe any single family's exact CPU/GPU split.</p>
 */
public enum DecoderOnlyRuntimeMode {

    /** Correctness-first Java reference implementation, entirely on the CPU. */
    REFERENCE("reference (CPU)"),

    /** WARP projection path: dense projections on the shared decoder-only WARP kernels, the rest on the CPU. */
    WARP("warp (WARP projection path)"),

    /** Prefer WARP and fall back to the reference implementation when the WARP device is unavailable. */
    AUTO("auto");

    private final String displayLabel;

    DecoderOnlyRuntimeMode(String displayLabel) {
        this.displayLabel = displayLabel;
    }

    /** Generic human-facing label; families may override with a more specific one. */
    public String displayLabel() {
        return displayLabel;
    }
}
