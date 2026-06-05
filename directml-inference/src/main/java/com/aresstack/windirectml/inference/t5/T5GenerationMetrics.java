package com.aresstack.windirectml.inference.t5;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Timing metrics for one T5 seq2seq generation run.
 *
 * <p>The values are intentionally stage-oriented rather than implementation-oriented.
 * They help decide which T5 stage should be moved further into WARP/DirectML without
 * leaking Qwen or decoder-only assumptions into the T5 runtime.</p>
 */
public final class T5GenerationMetrics {
    private static final T5GenerationMetrics EMPTY = new T5GenerationMetrics(
            0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0);

    private final long tokenizationNanos;
    private final long encoderNanos;
    private final long crossAttentionPrepareNanos;
    private final long decodeNanos;
    private final long lmHeadNanos;
    private final long tokenSelectionNanos;
    private final long detokenizationNanos;
    private final long runtimeNanos;
    private final int generatedTokens;

    public T5GenerationMetrics(long tokenizationNanos,
                               long encoderNanos,
                               long crossAttentionPrepareNanos,
                               long decodeNanos,
                               long lmHeadNanos,
                               long tokenSelectionNanos,
                               long detokenizationNanos,
                               long runtimeNanos,
                               int generatedTokens) {
        this.tokenizationNanos = nonNegative("tokenizationNanos", tokenizationNanos);
        this.encoderNanos = nonNegative("encoderNanos", encoderNanos);
        this.crossAttentionPrepareNanos = nonNegative("crossAttentionPrepareNanos", crossAttentionPrepareNanos);
        this.decodeNanos = nonNegative("decodeNanos", decodeNanos);
        this.lmHeadNanos = nonNegative("lmHeadNanos", lmHeadNanos);
        this.tokenSelectionNanos = nonNegative("tokenSelectionNanos", tokenSelectionNanos);
        this.detokenizationNanos = nonNegative("detokenizationNanos", detokenizationNanos);
        this.runtimeNanos = nonNegative("runtimeNanos", runtimeNanos);
        if (generatedTokens < 0) {
            throw new IllegalArgumentException("generatedTokens must not be negative: " + generatedTokens);
        }
        this.generatedTokens = generatedTokens;
    }

    public static T5GenerationMetrics empty() {
        return EMPTY;
    }

    public T5GenerationMetrics withTextBoundaryTimings(long tokenizationNanos, long detokenizationNanos) {
        return new T5GenerationMetrics(tokenizationNanos, encoderNanos, crossAttentionPrepareNanos, decodeNanos,
                lmHeadNanos, tokenSelectionNanos, detokenizationNanos, runtimeNanos, generatedTokens);
    }

    public long tokenizationNanos() {
        return tokenizationNanos;
    }

    public long encoderNanos() {
        return encoderNanos;
    }

    public long crossAttentionPrepareNanos() {
        return crossAttentionPrepareNanos;
    }

    public long decodeNanos() {
        return decodeNanos;
    }

    public long lmHeadNanos() {
        return lmHeadNanos;
    }

    public long tokenSelectionNanos() {
        return tokenSelectionNanos;
    }

    public long detokenizationNanos() {
        return detokenizationNanos;
    }

    public long runtimeNanos() {
        return runtimeNanos;
    }

    public int generatedTokens() {
        return generatedTokens;
    }

    public long tokenizationMillis() {
        return millis(tokenizationNanos);
    }

    public long encoderMillis() {
        return millis(encoderNanos);
    }

    public long crossAttentionPrepareMillis() {
        return millis(crossAttentionPrepareNanos);
    }

    public long decodeMillis() {
        return millis(decodeNanos);
    }

    public long lmHeadMillis() {
        return millis(lmHeadNanos);
    }

    public long tokenSelectionMillis() {
        return millis(tokenSelectionNanos);
    }

    public long detokenizationMillis() {
        return millis(detokenizationNanos);
    }

    public long runtimeMillis() {
        return millis(runtimeNanos);
    }

    public List<String> diagnosticLines() {
        if (this == EMPTY) {
            return Collections.emptyList();
        }
        List<String> lines = new ArrayList<String>();
        lines.add("  T5 profile runtime: " + runtimeMillis() + " ms");
        lines.add("    tokenize: " + tokenizationMillis() + " ms");
        lines.add("    encoder: " + encoderMillis() + " ms");
        lines.add("    cross-attn K/V prepare: " + crossAttentionPrepareMillis() + " ms");
        lines.add("    decoder steps: " + decodeMillis() + " ms");
        lines.add("    lm head: " + lmHeadMillis() + " ms");
        lines.add("    token select: " + tokenSelectionMillis() + " ms");
        lines.add("    detokenize: " + detokenizationMillis() + " ms");
        if (generatedTokens > 0) {
            lines.add("    avg/token runtime: " + millis(runtimeNanos / generatedTokens) + " ms");
        }
        return Collections.unmodifiableList(lines);
    }

    private static long nonNegative(String name, long value) {
        if (value < 0L) {
            throw new IllegalArgumentException(name + " must not be negative: " + value);
        }
        return value;
    }

    private static long millis(long nanos) {
        return nanos / 1_000_000L;
    }
}
