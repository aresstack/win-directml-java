package com.aresstack.windirectml.inference.gemma;

/**
 * Selects the Gemma 3 workbench execution path (GEMMA-WARP-11).
 *
 * <p>The default stays the {@link #EXTERNAL} local Python/Transformers probe; the native Java/WARP
 * runtime ({@link #NATIVE_WARP}) is experimental and only used when explicitly requested via
 * {@code -Dgemma.runtime=native-warp}. An explicit native flag never silently falls back to Python — it
 * either runs natively or fails with a clear message.</p>
 */
public enum Gemma3RuntimeMode {

    EXTERNAL("external-python-transformers"),
    NATIVE_WARP("native-warp-experimental");

    /** System property that selects the runtime: {@code external} (default) or {@code native-warp}. */
    public static final String PROPERTY = "gemma.runtime";

    private final String displayLabel;

    Gemma3RuntimeMode(String displayLabel) {
        this.displayLabel = displayLabel;
    }

    public String displayLabel() {
        return displayLabel;
    }

    /** Resolve the mode from {@code -Dgemma.runtime}; defaults to {@link #EXTERNAL} when unset/unknown. */
    public static Gemma3RuntimeMode fromSystemProperty() {
        return fromValue(System.getProperty(PROPERTY));
    }

    /** Parse a mode value: {@code native-warp} → {@link #NATIVE_WARP}, everything else → {@link #EXTERNAL}. */
    public static Gemma3RuntimeMode fromValue(String value) {
        if (value == null) {
            return EXTERNAL;
        }
        String v = value.trim().toLowerCase();
        return switch (v) {
            case "native-warp", "native_warp", "warp", "native" -> NATIVE_WARP;
            default -> EXTERNAL; // external / empty / unknown
        };
    }
}
