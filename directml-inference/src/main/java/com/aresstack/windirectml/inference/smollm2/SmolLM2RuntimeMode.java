package com.aresstack.windirectml.inference.smollm2;

/**
 * Selects the SmolLM2 execution path without changing model package semantics.
 */
public enum SmolLM2RuntimeMode {
    /**
     * Run the correctness-first Java reference implementation.
     */
    REFERENCE,

    /**
     * Require a future native/WARP executor and fail clearly when it is not available.
     */
    WARP,

    /**
     * Prefer the future native/WARP executor and fall back to the Java reference implementation.
     */
    AUTO
}
