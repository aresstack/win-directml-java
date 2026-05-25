package com.aresstack.windirectml.runtime;

/**
 * Execution provider / backend selection for the ML runtime.
 *
 * @see WinDirectMlRuntime.Builder#backend(Backend)
 */
public enum Backend {

    /**
     * Try DirectML first; fall back to CPU silently if DirectML is unavailable.
     */
    AUTO,

    /**
     * Force CPU execution. Fail visibly if the model cannot be loaded.
     */
    CPU,

    /**
     * Force DirectML (GPU) execution. Fail visibly if DirectML is unavailable.
     */
    DIRECTML;

    /**
     * Stable lowercase token for log/health output.
     */
    public String token() {
        return switch (this) {
            case AUTO -> "auto";
            case CPU -> "cpu";
            case DIRECTML -> "directml";
        };
    }
}
