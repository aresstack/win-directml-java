package com.aresstack.windirectml.runtime.api;

import com.aresstack.windirectml.encoder.e5.E5Variant;

/**
 * Strong public identifiers for embedding models supported by the Java 21 runtime API.
 */
public enum EmbeddingModelId {
    MINILM_L6_V2("all-MiniLM-L6-v2", "minilm", null, null),
    E5_SMALL_V2("e5-small-v2", "e5", E5Variant.SMALL_V2, "query: "),
    E5_BASE_V2("e5-base-v2", "e5", E5Variant.BASE_V2, "query: "),
    E5_LARGE_V2("e5-large-v2", "e5", E5Variant.LARGE_V2, "query: ");

    private final String directoryName;
    private final String family;
    private final E5Variant e5Variant;
    private final String defaultPrefix;

    EmbeddingModelId(String directoryName, String family, E5Variant e5Variant, String defaultPrefix) {
        this.directoryName = directoryName;
        this.family = family;
        this.e5Variant = e5Variant;
        this.defaultPrefix = defaultPrefix;
    }

    /** Default model directory name used by the workbench/download scripts. */
    public String directoryName() {
        return directoryName;
    }

    String family() {
        return family;
    }

    E5Variant e5Variant() {
        return e5Variant;
    }

    String defaultPrefix() {
        return defaultPrefix;
    }
}
