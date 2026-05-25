package com.aresstack.windirectml.runtime.facade;

import java.util.Locale;

/**
 * Hardware backend selection for the local ML runtime.
 * <p>
 * This controls which execution provider is used when loading models:
 * <ul>
 *   <li>{@link #AUTO} – try DirectML first, fall back to CPU if unavailable.</li>
 *   <li>{@link #DIRECTML} – require DirectML; fail if unavailable.</li>
 *   <li>{@link #CPU} – always use the CPU backend.</li>
 * </ul>
 */
public enum Backend {
    AUTO,
    DIRECTML,
    CPU;

    /**
     * Parse a backend string (case-insensitive). {@code null}/blank → {@link #AUTO}.
     *
     * @throws IllegalArgumentException for unknown values.
     */
    public static Backend parse(String raw) {
        if (raw == null || raw.isBlank()) return AUTO;
        return switch (raw.trim().toLowerCase(Locale.ROOT)) {
            case "auto" -> AUTO;
            case "directml", "dml" -> DIRECTML;
            case "cpu" -> CPU;
            default -> throw new IllegalArgumentException(
                    "Unknown backend: '" + raw + "' (expected: auto, directml, cpu)");
        };
    }
}
