package com.aresstack.windirectml.sidecar;

/**
 * Mutable runtime status of the sidecar process.
 * <p>
 * Reported through the {@code health} method and consulted by handlers that
 * must reject requests while the sidecar is starting up or shutting down.
 * All access is via {@code volatile} fields; updates happen on the main
 * dispatch thread.
 */
public final class SidecarStatus {

    private volatile boolean modelLoaded = false;
    private volatile boolean shuttingDown = false;
    private volatile boolean busy = false;
    private volatile String mode;
    private volatile String lastError;

    /**
     * Aktiver Embedding-Backend-Name ({@code "cpu"}, {@code "directml"},
     * {@code "custom"} oder {@code null} = keiner registriert).
     */
    private volatile String embeddingBackend;
    private volatile boolean embeddingReady = false;

    /**
     * Aktiver Reranker-Backend-Name ({@code "cpu"}, {@code "directml"},
     * {@code "custom"} oder {@code null} = keiner registriert).
     */
    private volatile String rerankerBackend;
    private volatile boolean rerankerReady = false;
    private volatile String rerankerModel;

    /**
     * {@code true} when the Phi-3 summarizer has finished loading and is ready
     * to process requests. Set from the phi3-model-loader thread.
     */
    private volatile boolean summarizerReady = false;

    /**
     * Active backend token for the Phi-3 summarizer ({@code "directml"},
     * {@code "cpu"}, or {@code null} = not loaded yet / not configured).
     */
    private volatile String summarizerBackend;

    /**
     * Human-readable model identifier reported in {@code health} once the
     * summarizer is ready, e.g. {@code "phi-3-mini-int4-directml"}.
     */
    private volatile String summarizerModel;

    public boolean isReady() {
        return modelLoaded && !shuttingDown;
    }

    public boolean isModelLoaded() {
        return modelLoaded;
    }

    public boolean isShuttingDown() {
        return shuttingDown;
    }

    public boolean isBusy() {
        return busy;
    }

    public String getMode() {
        return mode;
    }

    public String getLastError() {
        return lastError;
    }

    public void setModelLoaded(boolean modelLoaded) {
        this.modelLoaded = modelLoaded;
    }

    public void setShuttingDown(boolean shuttingDown) {
        this.shuttingDown = shuttingDown;
    }

    public void setBusy(boolean busy) {
        this.busy = busy;
    }

    public void setMode(String mode) {
        this.mode = mode;
    }

    public void setLastError(String lastError) {
        this.lastError = lastError;
    }

    public String getEmbeddingBackend() {
        return embeddingBackend;
    }

    public void setEmbeddingBackend(String embeddingBackend) {
        this.embeddingBackend = embeddingBackend;
    }

    public boolean isEmbeddingReady() {
        return embeddingReady;
    }

    public void setEmbeddingReady(boolean embeddingReady) {
        this.embeddingReady = embeddingReady;
    }

    public String getRerankerBackend() { return rerankerBackend; }
    public void setRerankerBackend(String rerankerBackend) { this.rerankerBackend = rerankerBackend; }
    public boolean isRerankerReady() { return rerankerReady; }
    public void setRerankerReady(boolean rerankerReady) { this.rerankerReady = rerankerReady; }
    public String getRerankerModel() { return rerankerModel; }
    public void setRerankerModel(String rerankerModel) { this.rerankerModel = rerankerModel; }

    public boolean isSummarizerReady() { return summarizerReady; }
    public void setSummarizerReady(boolean v) { this.summarizerReady = v; }

    public String getSummarizerBackend() { return summarizerBackend; }
    public void setSummarizerBackend(String v) { this.summarizerBackend = v; }

    public String getSummarizerModel() { return summarizerModel; }
    public void setSummarizerModel(String v) { this.summarizerModel = v; }
}
