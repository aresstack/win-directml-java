package com.aresstack.windirectml.inference.t5;

/**
 * Mutable metrics accumulator used during one T5 generation run.
 */
final class T5GenerationMetricsCollector {
    private long encoderNanos;
    private long crossAttentionPrepareNanos;
    private long decodeNanos;
    private long lmHeadNanos;
    private long tokenSelectionNanos;

    void addEncoderNanos(long nanos) {
        encoderNanos += requireNonNegative(nanos);
    }

    void addCrossAttentionPrepareNanos(long nanos) {
        crossAttentionPrepareNanos += requireNonNegative(nanos);
    }

    void addDecodeNanos(long nanos) {
        decodeNanos += requireNonNegative(nanos);
    }

    void addLmHeadNanos(long nanos) {
        lmHeadNanos += requireNonNegative(nanos);
    }

    void addTokenSelectionNanos(long nanos) {
        tokenSelectionNanos += requireNonNegative(nanos);
    }

    T5GenerationMetrics finish(long runtimeNanos, int generatedTokens) {
        return new T5GenerationMetrics(0L, encoderNanos, crossAttentionPrepareNanos, decodeNanos,
                lmHeadNanos, tokenSelectionNanos, 0L, runtimeNanos, generatedTokens);
    }

    private static long requireNonNegative(long nanos) {
        if (nanos < 0L) {
            throw new IllegalArgumentException("nanos must not be negative: " + nanos);
        }
        return nanos;
    }
}
