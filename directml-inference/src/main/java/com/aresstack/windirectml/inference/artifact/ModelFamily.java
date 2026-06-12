package com.aresstack.windirectml.inference.artifact;

/**
 * Model families known to the unified artifact lifecycle.
 *
 * <p>Families with a real runtime-package compiler ({@link #QWEN}, {@link #SMOLLM2},
 * {@link #T5}) flow through {@code download -> validate raw -> convert -> validate
 * package -> run only from package}. Families without a {@code .wdmlpack} compiler yet
 * ({@link #EMBEDDING}, {@link #RERANKER}, {@link #PHI3}) are explicitly marked as a
 * direct-source legacy path rather than being hidden.</p>
 */
public enum ModelFamily {

    QWEN("Qwen2.5-Coder"),
    SMOLLM2("SmolLM2"),
    T5("T5/CodeT5"),
    EMBEDDING("Embedding"),
    RERANKER("Reranker"),
    PHI3("Phi-3");

    private final String displayName;

    ModelFamily(String displayName) {
        this.displayName = displayName;
    }

    public String displayName() {
        return displayName;
    }
}
