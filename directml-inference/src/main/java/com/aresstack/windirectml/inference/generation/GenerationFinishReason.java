package com.aresstack.windirectml.inference.generation;

/**
 * Model-family-neutral generation finish reason.
 *
 * <p>Shared vocabulary so callers (workbench/API) can interpret why generation ended without knowing whether it ran on
 * the decoder-only or the seq2seq (T5) runtime. Each family keeps its own finish-reason type (e.g. T5's
 * {@code T5RuntimeResult.FinishReason} enum, decoder-only's {@code "eos_token"}/{@code "length"} string) and maps onto
 * this one — the family types are NOT renamed.</p>
 */
public enum GenerationFinishReason {

    /** Generation ended because a stop/eos token was selected. */
    STOP_TOKEN,

    /** Generation ended because the maximum number of new tokens was reached. */
    LENGTH,

    /** Generation could not run (unsupported runtime/path). */
    UNSUPPORTED;

    /**
     * Map a decoder-only finish-reason string ({@code "eos_token"} / {@code "length"}) onto the shared enum. Pure
     * string mapping; introduces no dependency on the decoder-only package.
     */
    public static GenerationFinishReason fromDecoderOnlyReason(String reason) {
        if ("eos_token".equals(reason)) {
            return STOP_TOKEN;
        }
        if ("length".equals(reason)) {
            return LENGTH;
        }
        throw new IllegalArgumentException("Unknown decoder-only finish reason: " + reason);
    }
}
