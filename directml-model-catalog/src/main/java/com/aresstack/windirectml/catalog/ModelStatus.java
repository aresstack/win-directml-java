package com.aresstack.windirectml.catalog;

import java.util.Locale;

/**
 * Runtime-support status of a catalog entry. Only {@link #RUNNABLE} entries may be offered as a local
 * install recommendation and consumed as a productive local model. {@link #UNVERIFIED} is a deliberate
 * middle state for a checkpoint whose runtime path exists but has not yet passed a real compile + load +
 * task smoke (e.g. a reranker known only by an enum directory-name placeholder): it is neither advertised
 * as runnable nor silently promoted.
 *
 * <p>Java-8 compatible.</p>
 */
public enum ModelStatus {

    /** Proven end to end (compile + smoke-load + real task) on at least one supported backend. */
    RUNNABLE,
    /** A runtime path exists but real verification is still missing; NOT offered as a local recommendation. */
    UNVERIFIED,
    /** Metadata only; no runtime path in this build. */
    PLANNED,
    /** Explicitly not supported by this project. */
    UNSUPPORTED;

    public String token() {
        return name().toLowerCase(Locale.ROOT);
    }

    /** Whether a host may offer this entry as a local install recommendation and run it productively. */
    public boolean isRunnable() {
        return this == RUNNABLE;
    }
}
