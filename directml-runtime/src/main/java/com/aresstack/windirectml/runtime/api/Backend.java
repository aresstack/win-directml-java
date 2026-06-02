package com.aresstack.windirectml.runtime.api;

/**
 * Public backend selection for local inference.
 */
public enum Backend {
    /**
     * Try DirectML first and fall back to CPU when necessary.
     */
    AUTO,
    /**
     * Force CPU execution.
     */
    CPU,
    /**
     * Force Windows DirectML execution.
     */
    DIRECTML;

    com.aresstack.windirectml.runtime.facade.Backend toFacade() {
        return com.aresstack.windirectml.runtime.facade.Backend.valueOf(name());
    }

    static Backend fromFacade(com.aresstack.windirectml.runtime.facade.Backend backend) {
        return Backend.valueOf(backend.name());
    }
}
