package com.aresstack.windirectml.inference.smollm2;

/**
 * Accumulates low-level timings from the SmolLM2 reference forward pass.
 */
final class SmolLM2ReferenceForwardProfile {

    private long layerNormNanos;
    private long attentionProjectionNanos;
    private long attentionScoreNanos;
    private long attentionOutputProjectionNanos;
    private long mlpNanos;
    private long finalNormNanos;
    private long lmHeadNanos;

    void addLayerNormNanos(long nanos) {
        layerNormNanos += Math.max(0L, nanos);
    }

    void addAttentionProjectionNanos(long nanos) {
        attentionProjectionNanos += Math.max(0L, nanos);
    }

    void addAttentionScoreNanos(long nanos) {
        attentionScoreNanos += Math.max(0L, nanos);
    }

    void addAttentionOutputProjectionNanos(long nanos) {
        attentionOutputProjectionNanos += Math.max(0L, nanos);
    }

    void addMlpNanos(long nanos) {
        mlpNanos += Math.max(0L, nanos);
    }

    void addFinalNormNanos(long nanos) {
        finalNormNanos += Math.max(0L, nanos);
    }

    void addLmHeadNanos(long nanos) {
        lmHeadNanos += Math.max(0L, nanos);
    }

    long lmHeadNanos() {
        return lmHeadNanos;
    }

    SmolLM2ReferenceHotspotProfile snapshot() {
        return new SmolLM2ReferenceHotspotProfile(
                layerNormNanos,
                attentionProjectionNanos,
                attentionScoreNanos,
                attentionOutputProjectionNanos,
                mlpNanos,
                finalNormNanos,
                lmHeadNanos);
    }
}
