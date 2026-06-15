package com.aresstack.windirectml.inference.gemma;

/**
 * Phase + WARP-counter profile of one native Java/WARP Gemma 3 generation (GEMMA-WORKBENCH-PROFILING-1).
 *
 * <p>Measured inside {@link Gemma3NativeWarpRuntime}: the load phases (package open, tokenizer load,
 * weight load, WARP/session init) and the generation phases (tokenize, prefill = time to first token,
 * decode total, detokenize), plus the {@link com.aresstack.windirectml.windows.WarpSubmissionStats}
 * deltas over the generate call (submits / fence waits / readbacks). All durations are milliseconds.
 * The prompt-template render time and the run's grand total are panel-level and supplied to
 * {@link Gemma3NativeWarpProfileReport} separately.</p>
 */
public record Gemma3NativeWarpProfile(
        long packageOpenMs,
        long tokenizerLoadMs,
        long weightLoadMs,
        long sessionInitMs,
        long tokenizeMs,
        long prefillMs,
        long decodeTotalMs,
        long detokenizeMs,
        long runtimeTotalMs,
        int promptTokens,
        int outputTokens,
        long submits,
        long fenceWaits,
        long readbacks) {

    /** Sum of the four load phases. */
    public long loadTotalMs() {
        return packageOpenMs + tokenizerLoadMs + weightLoadMs + sessionInitMs;
    }

    /**
     * Average decode time per decoded token. The first token comes from prefill, so this divides the
     * decode total by the number of {@code decodeNext} steps ({@code outputTokens - 1}); 0 when there is
     * at most one output token.
     */
    public double decodeAvgPerTokenMs() {
        int steps = outputTokens - 1;
        return steps > 0 ? (double) decodeTotalMs / steps : 0.0;
    }

    public double submitsPerToken() {
        return outputTokens > 0 ? (double) submits / outputTokens : 0.0;
    }

    public double fenceWaitsPerToken() {
        return outputTokens > 0 ? (double) fenceWaits / outputTokens : 0.0;
    }

    public double readbacksPerToken() {
        return outputTokens > 0 ? (double) readbacks / outputTokens : 0.0;
    }
}
