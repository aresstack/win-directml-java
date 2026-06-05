package com.aresstack.windirectml.inference.smollm2;

/**
 * Logical buffer categories prepared for the future SmolLM2 WARP runtime.
 */
public enum SmolLM2WarpBufferKind {
    WEIGHT,
    KV_CACHE,
    SCRATCH
}
