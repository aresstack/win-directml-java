package com.aresstack.windirectml.config.generation;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Workbench generation output mode (GENERATION-STREAMING-1): whether generated tokens are shown live as
 * they are produced ({@link #STREAMING}) or collected and shown once at the end ({@link #BUFFERED}).
 *
 * <p>The default is always {@link #STREAMING}. It can be overridden at startup via
 * {@code -Ddirectml.generation.output=streaming|buffered} or the boolean
 * {@code -Ddirectml.generation.streaming=true|false}; unset/empty/unknown values fall back to STREAMING.
 * This is family-neutral — every generation model in the Workbench respects the same switch.</p>
 *
 * <p>This module targets Java 8, so this type avoids switch expressions and {@code String.isBlank()}.</p>
 */
public enum GenerationOutputMode {

    STREAMING,
    BUFFERED;

    /** {@code -Ddirectml.generation.output=streaming|buffered}. */
    public static final String OUTPUT_PROPERTY = "directml.generation.output";
    /** {@code -Ddirectml.generation.streaming=true|false}. */
    public static final String STREAMING_PROPERTY = "directml.generation.streaming";

    private static final Set<String> BUFFERED_OUTPUT_VALUES = new HashSet<String>(
            Arrays.asList("buffered", "non-streaming", "nonstreaming", "false", "off"));
    private static final Set<String> BUFFERED_BOOLEAN_VALUES = new HashSet<String>(
            Arrays.asList("false", "off", "0", "no"));

    public boolean isStreaming() {
        return this == STREAMING;
    }

    /** Resolve from the system properties; defaults to {@link #STREAMING}. */
    public static GenerationOutputMode fromSystemProperty() {
        String output = System.getProperty(OUTPUT_PROPERTY);
        if (output != null && !output.trim().isEmpty()) {
            return fromOutputValue(output);
        }
        String streaming = System.getProperty(STREAMING_PROPERTY);
        if (streaming != null && !streaming.trim().isEmpty()) {
            return fromStreamingValue(streaming);
        }
        return STREAMING;
    }

    /** Parse a {@code directml.generation.output} value; only buffered/non-streaming/false/off → BUFFERED. */
    public static GenerationOutputMode fromOutputValue(String value) {
        if (value == null) {
            return STREAMING;
        }
        return BUFFERED_OUTPUT_VALUES.contains(value.trim().toLowerCase()) ? BUFFERED : STREAMING;
    }

    /** Parse a {@code directml.generation.streaming} boolean value; only an explicit false/off/0/no → BUFFERED. */
    public static GenerationOutputMode fromStreamingValue(String value) {
        if (value == null) {
            return STREAMING;
        }
        return BUFFERED_BOOLEAN_VALUES.contains(value.trim().toLowerCase()) ? BUFFERED : STREAMING;
    }
}
