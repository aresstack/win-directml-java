package com.aresstack.windirectml.inference.gemma;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Opt-in, coarse per-kernel-group timing accumulator for the WARP/DirectML Gemma decode path
 * (GEMMA-WARP-17). It aggregates wall-clock time and recorded-dispatch counts per named pipeline group
 * (QKV projection, attention, GeGLU, …) across several warm decode tokens, so the next WARP bottleneck can
 * be chosen from measured group shares rather than dispatch counts alone.
 *
 * <p>Measurement model: when a profiler is attached to a {@link com.aresstack.windirectml.windows.WarpExecutionContext}
 * the decode step runs <b>synchronously</b> (no command-list coalescing / deferred batch), so each group's
 * kernels execute and fence before the next group starts and the wall-clock between {@code mark()} boundaries
 * is that group's GPU time (plus its submit overhead). This serializes the pipeline, so the absolute
 * ms/token here is <em>higher</em> than the real coalesced decode — use the per-group <b>percentages</b>
 * together with the separately measured real decode avg/token. The normal runtime never attaches a profiler
 * and is unaffected.</p>
 */
public final class WarpGroupProfiler {

    /** Aggregated, per-decode-token figures for one group. */
    public record GroupStat(String name, double callsPerToken, double dispatchesPerToken,
                            double barriersPerToken, double msPerToken, double avgMsPerCall, double percent) {
    }

    private static final class Acc {
        long calls;
        long dispatches;
        long nanos;
    }

    private final Map<String, Acc> groups = new LinkedHashMap<>();
    private int tokens;

    /** Record one group invocation: its recorded-dispatch delta and elapsed nanos. */
    public void record(String name, long dispatches, long nanos) {
        Acc a = groups.computeIfAbsent(name, n -> new Acc());
        a.calls++;
        a.dispatches += dispatches;
        a.nanos += nanos;
    }

    /** Mark the end of one decoded token (for per-token averaging). */
    public void incTokens() {
        tokens++;
    }

    public int tokens() {
        return tokens;
    }

    /**
     * Per-token group breakdown, largest time share first. {@code barriersPerToken} equals
     * {@code dispatchesPerToken} (the coalesced product path records exactly one UAV barrier per dispatch),
     * so the count column doubles as the barrier column.
     */
    public List<GroupStat> perToken() {
        long totalNanos = groups.values().stream().mapToLong(a -> a.nanos).sum();
        int t = Math.max(tokens, 1);
        List<GroupStat> out = new ArrayList<>();
        for (Map.Entry<String, Acc> e : groups.entrySet()) {
            Acc a = e.getValue();
            double callsPerTok = a.calls / (double) t;
            double dispPerTok = a.dispatches / (double) t;
            double msPerTok = a.nanos / 1e6 / t;
            double avgMs = a.calls == 0 ? 0 : a.nanos / 1e6 / a.calls;
            double pct = totalNanos == 0 ? 0 : 100.0 * a.nanos / totalNanos;
            out.add(new GroupStat(e.getKey(), callsPerTok, dispPerTok, dispPerTok, msPerTok, avgMs, pct));
        }
        out.sort((x, y) -> Double.compare(y.percent(), x.percent()));
        return out;
    }

    /** Total measured (serialized) ms per token across all groups. */
    public double totalMsPerToken() {
        int t = Math.max(tokens, 1);
        return groups.values().stream().mapToLong(a -> a.nanos).sum() / 1e6 / t;
    }

    /** A human-readable table. {@code realDecodeMsPerToken} is the separately measured coalesced decode time. */
    public List<String> render(double realDecodeMsPerToken) {
        List<String> lines = new ArrayList<>();
        lines.add(String.format(Locale.ROOT,
                "Gemma WARP decode group timing (%d warm tokens; serialized profile total=%.1f ms/token; "
                        + "real coalesced decode=%.1f ms/token):", tokens, totalMsPerToken(), realDecodeMsPerToken));
        lines.add(String.format(Locale.ROOT, "  %-26s %8s %12s %11s %11s %9s %7s",
                "group", "calls/tok", "dispatch/tok", "barrier/tok", "ms/token", "ms/call", "%"));
        for (GroupStat g : perToken()) {
            lines.add(String.format(Locale.ROOT, "  %-26s %8.1f %12.1f %11.1f %11.2f %9.3f %6.1f%%",
                    g.name(), g.callsPerToken(), g.dispatchesPerToken(), g.barriersPerToken(),
                    g.msPerToken(), g.avgMsPerCall(), g.percent()));
        }
        return lines;
    }
}
