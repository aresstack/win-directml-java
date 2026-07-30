package com.aresstack.windirectml.inference.api;

/**
 * Why generation stopped. Part of the stable public API (used by {@link GenerationResult}); the
 * internal engines keep their own finish-reason types and map onto this one, so no internal type
 * leaks through the public surface.
 */
public enum GenerationFinishReason {

    /** A stop / end-of-sequence token was produced (natural completion). */
    STOP,

    /** The maximum number of new tokens was reached. */
    LENGTH,

    /** Generation was cancelled by the caller. */
    CANCELLED,

    /** Generation ended abnormally. */
    ERROR
}
