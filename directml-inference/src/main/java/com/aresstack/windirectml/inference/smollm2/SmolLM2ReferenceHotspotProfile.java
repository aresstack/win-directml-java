package com.aresstack.windirectml.inference.smollm2;

/**
 * Captures accumulated reference-runtime hotspot timings for SmolLM2.
 */
public record SmolLM2ReferenceHotspotProfile(long layerNormNanos,
                                             long attentionProjectionNanos,
                                             long attentionScoreNanos,
                                             long attentionOutputProjectionNanos,
                                             long mlpNanos,
                                             long finalNormNanos,
                                             long lmHeadNanos) {

    public SmolLM2ReferenceHotspotProfile {
        layerNormNanos = nonNegative(layerNormNanos);
        attentionProjectionNanos = nonNegative(attentionProjectionNanos);
        attentionScoreNanos = nonNegative(attentionScoreNanos);
        attentionOutputProjectionNanos = nonNegative(attentionOutputProjectionNanos);
        mlpNanos = nonNegative(mlpNanos);
        finalNormNanos = nonNegative(finalNormNanos);
        lmHeadNanos = nonNegative(lmHeadNanos);
    }

    public static SmolLM2ReferenceHotspotProfile empty() {
        return new SmolLM2ReferenceHotspotProfile(
                0L,
                0L,
                0L,
                0L,
                0L,
                0L,
                0L);
    }

    public long measuredMillis() {
        return nanosToMillis(
                layerNormNanos
                        + attentionProjectionNanos
                        + attentionScoreNanos
                        + attentionOutputProjectionNanos
                        + mlpNanos
                        + finalNormNanos
                        + lmHeadNanos);
    }

    public long layerNormMillis() {
        return nanosToMillis(layerNormNanos);
    }

    public long attentionProjectionMillis() {
        return nanosToMillis(attentionProjectionNanos);
    }

    public long attentionScoreMillis() {
        return nanosToMillis(attentionScoreNanos);
    }

    public long attentionOutputProjectionMillis() {
        return nanosToMillis(attentionOutputProjectionNanos);
    }

    public long mlpMillis() {
        return nanosToMillis(mlpNanos);
    }

    public long finalNormMillis() {
        return nanosToMillis(finalNormNanos);
    }

    public long lmHeadMillis() {
        return nanosToMillis(lmHeadNanos);
    }

    private static long nonNegative(long nanos) {
        return Math.max(0L, nanos);
    }

    private static long nanosToMillis(long nanos) {
        return nanos / 1_000_000L;
    }
}