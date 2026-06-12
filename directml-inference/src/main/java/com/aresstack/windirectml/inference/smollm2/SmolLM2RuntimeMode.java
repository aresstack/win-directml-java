package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyRuntimeMode;

/**
 * Selects the SmolLM2 execution path without changing model package semantics.
 */
public enum SmolLM2RuntimeMode {
    /**
     * Run the correctness-first Java reference implementation entirely on the CPU.
     */
    REFERENCE("reference (CPU)"),

    /**
     * WARP projection path: every dense projection (q/k/v/o, gate/up/down, lm_head) runs on the shared decoder-only
     * WARP kernels, while norms, RoPE, attention scoring/context, SwiGLU and the KV cache still run on the CPU.
     * This is a WARP-assisted decoder-only runtime, not a fully GPU-resident decode loop.
     */
    WARP("warp (WARP projection path; norms/RoPE/attention/KV-cache on CPU)"),

    /**
     * Prefer the WARP projection path and fall back to the Java reference implementation when the WARP device or
     * weight upload is unavailable.
     */
    AUTO("auto");

    private final String displayLabel;

    SmolLM2RuntimeMode(String displayLabel) {
        this.displayLabel = displayLabel;
    }

    /**
     * Human-facing label that honestly describes what the mode runs on (for diagnostics/UI output).
     */
    public String displayLabel() {
        return displayLabel;
    }

    /**
     * Map onto the family-neutral {@link DecoderOnlyRuntimeMode}. SmolLM2 keeps its own, more specific
     * {@link #displayLabel()}; this is the adapter point a shared decoder-only runtime selector consumes.
     */
    public DecoderOnlyRuntimeMode toDecoderOnlyRuntimeMode() {
        return switch (this) {
            case REFERENCE -> DecoderOnlyRuntimeMode.REFERENCE;
            case WARP -> DecoderOnlyRuntimeMode.WARP;
            case AUTO -> DecoderOnlyRuntimeMode.AUTO;
        };
    }
}
