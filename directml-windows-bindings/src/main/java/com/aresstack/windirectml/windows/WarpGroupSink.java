package com.aresstack.windirectml.windows;

/**
 * Opt-in sink for coarse per-group decode timing (GEMMA-WARP-17). A {@link WarpExecutionContext} with a sink
 * attached runs synchronously and reports, at each {@link WarpExecutionContext#mark} boundary, the previous
 * group's name, its recorded-dispatch delta and elapsed nanoseconds. Implemented by the inference-layer
 * group profiler (kept here as a tiny interface so the binding layer does not depend on inference).
 */
@FunctionalInterface
public interface WarpGroupSink {

    /** Record one completed group: its recorded-dispatch count and wall-clock nanoseconds. */
    void record(String group, long dispatches, long nanos);
}
