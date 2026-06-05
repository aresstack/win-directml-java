package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * Scoped profiler for T5 internals that are not visible from the generation loop.
 */
final class T5GenerationProfiler {
    private static final ThreadLocal<T5GenerationMetricsCollector> CURRENT = new ThreadLocal<T5GenerationMetricsCollector>();

    private T5GenerationProfiler() {
    }

    static Scope open(T5GenerationMetricsCollector collector) {
        Objects.requireNonNull(collector, "collector");
        T5GenerationMetricsCollector previous = CURRENT.get();
        CURRENT.set(collector);
        return new Scope(previous);
    }

    static void recordCrossAttentionPrepareNanos(long nanos) {
        T5GenerationMetricsCollector collector = CURRENT.get();
        if (collector != null) {
            collector.addCrossAttentionPrepareNanos(nanos);
        }
    }

    static final class Scope implements AutoCloseable {
        private final T5GenerationMetricsCollector previous;
        private boolean closed;

        private Scope(T5GenerationMetricsCollector previous) {
            this.previous = previous;
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                if (previous == null) {
                    CURRENT.remove();
                } else {
                    CURRENT.set(previous);
                }
            }
        }
    }
}
