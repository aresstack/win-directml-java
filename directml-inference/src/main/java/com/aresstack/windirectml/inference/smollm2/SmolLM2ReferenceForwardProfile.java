package com.aresstack.windirectml.inference.smollm2;

/**
 * Accumulates low-level timings from the SmolLM2 reference forward pass.
 */
final class SmolLM2ReferenceForwardProfile {

    private long lmHeadNanos;

    void addLmHeadNanos(long nanos) {
        lmHeadNanos += Math.max(0L, nanos);
    }

    long lmHeadNanos() {
        return lmHeadNanos;
    }
}
