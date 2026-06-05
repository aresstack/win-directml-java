package com.aresstack.windirectml.inference.t5;

/**
 * T5 parallel-strand payload policy.
 */
final class T5ManifestPayloadPolicy {
    static final boolean PAYLOAD_INCLUDED = false;
    static final boolean RUNTIME_LOADABLE = false;
    static final String RUNTIME_LOAD_MODE = "not-implemented";
    static final String REASON = "T5 runtime is not implemented yet";

    private T5ManifestPayloadPolicy() {
    }
}
