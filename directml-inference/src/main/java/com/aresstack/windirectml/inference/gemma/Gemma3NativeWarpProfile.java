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
 *
 * <p>The {@code submits}/{@code fenceWaits}/{@code readbacks} are the totals over the whole generate
 * region (prefill + decode). The {@code decode*} counters cover only the decode region (after the first
 * token), so the per-token figures are the <b>decode steady-state</b> the perf slices report (≈93 fence
 * waits, ≈37 readbacks/token after 13b-3b) rather than a prefill-amortised average.</p>
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
        long readbacks,
        long decodeSubmits,
        long decodeFenceWaits,
        long decodeReadbacks,
        long decodeDispatches,
        long decodeUavBarriers,
        Gemma3WarpExecutionMode executionMode,
        String adapterDescription,
        boolean adapterSoftware) {

    /** Sum of the four load phases. */
    public long loadTotalMs() {
        return packageOpenMs + tokenizerLoadMs + weightLoadMs + sessionInitMs;
    }

    /** Number of {@code decodeNext} steps (the first token comes from prefill). */
    public int decodeSteps() {
        return Math.max(0, outputTokens - 1);
    }

    /**
     * Average decode time per decoded token: the decode total over the number of {@code decodeNext}
     * steps; 0 when there is at most one output token.
     */
    public double decodeAvgPerTokenMs() {
        return decodeSteps() > 0 ? (double) decodeTotalMs / decodeSteps() : 0.0;
    }

    public double submitsPerToken() {
        return decodeSteps() > 0 ? (double) decodeSubmits / decodeSteps() : 0.0;
    }

    public double fenceWaitsPerToken() {
        return decodeSteps() > 0 ? (double) decodeFenceWaits / decodeSteps() : 0.0;
    }

    public double readbacksPerToken() {
        return decodeSteps() > 0 ? (double) decodeReadbacks / decodeSteps() : 0.0;
    }

    public double dispatchesPerToken() {
        return decodeSteps() > 0 ? (double) decodeDispatches / decodeSteps() : 0.0;
    }

    public double uavBarriersPerToken() {
        return decodeSteps() > 0 ? (double) decodeUavBarriers / decodeSteps() : 0.0;
    }
}
