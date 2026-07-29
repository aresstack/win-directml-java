package com.aresstack.windirectml.catalog;

import java.util.Locale;

/**
 * The runtime family of a catalog entry: the unit that shares one wdmlpack compiler / package lifecycle and
 * one runtime adapter. Carries the canonical {@code .wdmlpack} package file name produced by that family's
 * compiler and a stable {@code lifecycleId} an install strategy dispatches on. Encoders and rerankers share
 * the BERT encoder stack but produce distinct package files, so they are distinct families here.
 *
 * <p>Java-8 compatible.</p>
 */
public enum CatalogModelFamily {

    MINILM("MiniLM embedding", "encoder", "encoder.wdmlpack"),
    E5("E5 embedding", "encoder", "encoder.wdmlpack"),
    CROSS_ENCODER("Cross-encoder reranker", "reranker", "reranker.wdmlpack"),
    QWEN("Qwen2.5-Coder", "qwen", "model_q4f16.wdmlpack"),
    SMOLLM2("SmolLM2", "smollm2", "model_smollm2.wdmlpack"),
    GEMMA3("Gemma 3", "gemma3", "model_gemma3.wdmlpack"),
    PHI3("Phi-3", "phi3", "model_phi3.wdmlpack"),
    T5("T5/CodeT5", "t5", "model_t5.wdmlpack");

    private final String displayName;
    private final String lifecycleId;
    private final String packageFileName;

    CatalogModelFamily(String displayName, String lifecycleId, String packageFileName) {
        this.displayName = displayName;
        this.lifecycleId = lifecycleId;
        this.packageFileName = packageFileName;
    }

    public String displayName() {
        return displayName;
    }

    /** Stable id an install strategy / package-lifecycle registry dispatches on. */
    public String lifecycleId() {
        return lifecycleId;
    }

    /** Canonical compiled runtime-package file name produced for this family. */
    public String packageFileName() {
        return packageFileName;
    }

    public String token() {
        return name().toLowerCase(Locale.ROOT);
    }
}
