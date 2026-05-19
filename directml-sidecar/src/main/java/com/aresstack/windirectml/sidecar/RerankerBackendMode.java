package com.aresstack.windirectml.sidecar;

import java.util.Locale;

/**
 * Backend selector for the cross-encoder reranker, parsed from the
 * {@code -Drerank.backend} system property.
 *
 * <p>Mirrors {@link EmbeddingBackendSelector.Mode} so {@code rerank.backend}
 * and {@code embed.backend} share the same UX:
 * <ul>
 *   <li>{@code cpu} – force the CPU reranker; missing model or failed
 *       load is fatal (exit 3).</li>
 *   <li>{@code directml} – force the DirectML reranker; missing model
 *       or failed DirectML init is fatal (exit 3).</li>
 *   <li>{@code auto} (default) – try DirectML first and fall back to CPU
 *       on failure. The fallback is surfaced through
 *       {@code health.rerankerBackend} and {@code health.lastError}.
 *       Missing model directory is non-fatal: the {@code rerank} handler
 *       stays in {@code not-implemented} mode.</li>
 * </ul>
 * Unknown values throw {@link IllegalArgumentException}; callers in
 * {@code main()} translate that to {@code System.exit(2)}.
 */
public enum RerankerBackendMode {
    CPU, DIRECTML, AUTO;

    /**
     * Parse a {@code -Drerank.backend} value. {@code null}/blank defaults
     * to {@link #AUTO}.
     */
    public static RerankerBackendMode parse(String raw) {
        if (raw == null || raw.isBlank()) return AUTO;
        return switch (raw.trim().toLowerCase(Locale.ROOT)) {
            case "cpu" -> CPU;
            case "directml", "dml" -> DIRECTML;
            case "auto" -> AUTO;
            default -> throw new IllegalArgumentException(
                    "Unknown rerank.backend: '" + raw
                            + "' (expected one of: cpu, directml, auto)");
        };
    }

    /**
     * Stable lowercase token used in health/log output.
     */
    public String token() {
        return switch (this) {
            case CPU -> "cpu";
            case DIRECTML -> "directml";
            case AUTO -> "auto";
        };
    }
}

