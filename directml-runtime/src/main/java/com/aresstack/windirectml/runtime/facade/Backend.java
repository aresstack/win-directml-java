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
 *   <li>{@link #HYBRID} – use DirectML for batched prefill, CPU for per-token decode.
 *       Currently only honoured by the Qwen2 engine; other engines treat this as
 *       {@link #AUTO}. Designed for Intel iGPU hosts where each DirectML fence-wait
 *       costs 10–40&nbsp;ms — running 24-layer per-token decode through the GPU
 *       costs ~1.5&nbsp;s/token of pure submission overhead, but a 30-token
 *       prefill in one batched GEMM amortises that to a few hundred ms total.</li>
 * </ul>
 */
public enum Backend {
    AUTO,
    DIRECTML,
    CPU,
    HYBRID;

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
            case "hybrid" -> HYBRID;
            default -> throw new IllegalArgumentException(
                    "Unknown backend: '" + raw + "' (expected: auto, directml, cpu, hybrid)");
        };
    }
}
