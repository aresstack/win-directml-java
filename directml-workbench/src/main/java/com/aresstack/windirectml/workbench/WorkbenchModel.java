package com.aresstack.windirectml.workbench;

import com.aresstack.windirectml.runtime.facade.Backend;

import java.nio.file.Path;

/**
 * Shared mutable state for workbench panels.
 */
public final class WorkbenchModel {

    private volatile Backend backend = Backend.AUTO;
    private volatile Path modelRoot = Path.of("model");
    private volatile String embeddingModel = "all-MiniLM-L6-v2";
    private volatile String rerankerModel = "cross-encoder-ms-marco-MiniLM-L-6-v2";
    private volatile String summarizerModel = "microsoft/Phi-3-mini-4k-instruct-onnx";

    public Backend getBackend() { return backend; }
    public void setBackend(Backend backend) { this.backend = backend; }

    public Path getModelRoot() { return modelRoot; }
    public void setModelRoot(Path modelRoot) { this.modelRoot = modelRoot; }

    public String getEmbeddingModel() { return embeddingModel; }
    public void setEmbeddingModel(String embeddingModel) { this.embeddingModel = embeddingModel; }

    public String getRerankerModel() { return rerankerModel; }
    public void setRerankerModel(String rerankerModel) { this.rerankerModel = rerankerModel; }

    public String getSummarizerModel() { return summarizerModel; }
    public void setSummarizerModel(String summarizerModel) { this.summarizerModel = summarizerModel; }
}
