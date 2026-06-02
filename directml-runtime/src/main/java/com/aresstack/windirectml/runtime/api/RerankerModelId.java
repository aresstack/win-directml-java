package com.aresstack.windirectml.runtime.api;

/**
 * Strong public identifiers for cross-encoder reranker models supported by the Java 21 runtime API.
 */
public enum RerankerModelId {
    MS_MARCO_MINILM_L6("cross-encoder-ms-marco-MiniLM-L-6-v2"),
    MS_MARCO_MINILM_L12("cross-encoder-ms-marco-MiniLM-L-12-v2"),
    BGE_RERANKER_BASE("bge-reranker-base");

    private final String directoryName;

    RerankerModelId(String directoryName) {
        this.directoryName = directoryName;
    }

    /**
     * Default model directory name used by the workbench/download scripts.
     */
    public String directoryName() {
        return directoryName;
    }
}
