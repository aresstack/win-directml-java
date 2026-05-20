package com.aresstack.windirectml.sidecar.client;

/**
 * Mutable POJO holding everything needed to spawn a sidecar process and
 * talk to it over JSON-RPC. Java-8 compatible.
 *
 * <p>Typical usage:
 * <pre>
 *   SidecarClientConfig cfg = new SidecarClientConfig();
 *   cfg.setJavaExecutable("C:\\Program Files\\Java\\jdk-21\\bin\\java.exe");
 *   cfg.setSidecarJarPath("directml-sidecar.jar");
 *   cfg.setModelDirectory("model/all-MiniLM-L6-v2");
 *   cfg.setEmbedBackend("directml");
 *   SidecarClient client = new SidecarClient(cfg);
 *   client.start();
 * </pre>
 */
public final class SidecarClientConfig {

    private String javaExecutable = "java";
    private String sidecarJarPath;
    private String modelDirectory;
    /**
     * {@code auto} (default), {@code directml}, {@code cpu}.
     */
    private String embedBackend = "auto";
    /**
     * Encoder family: {@code minilm} (default) or {@code e5}.
     * Forwarded to the sidecar as {@code -Dembed.model=<value>}.
     */
    private String embedModel = "minilm";
    /**
     * E5 variant token when {@link #embedModel} is {@code e5}:
     * {@code small-v2}, {@code base-v2}, {@code large-v2}, or
     * {@code base-sts-en-de} (default). Forwarded as
     * {@code -De5.model=<value>}.
     */
    private String e5Variant = "base-sts-en-de";
    /**
     * Filesystem path to the E5 model directory. Forwarded as
     * {@code -De5.modelDir=<value>}. When blank, the sidecar
     * auto-discovers from the variant's directory hints (e.g.
     * {@code model/e5-base-sts-en-de}).
     */
    private String e5ModelDirectory;
    /**
     * Backend selector for the cross-encoder reranker, forwarded as
     * {@code -Drerank.backend=<value>}. Accepted values:
     * {@code auto} (default), {@code directml}, {@code cpu}. Unknown
     * values cause the sidecar to exit with code 2.
     */
    private String rerankBackend = "auto";
    /**
     * Filesystem path to the cross-encoder reranker model directory,
     * forwarded as {@code -Drerank.modelDir=<value>}. When blank, the
     * sidecar auto-discovers conventional locations under {@code model/}
     * (e.g. {@code model/cross-encoder-ms-marco-MiniLM-L-6-v2}).
     */
    private String rerankModelDirectory;
    private boolean directmlDebug = false;
    /**
     * Optional override DLL path forwarded as {@code -Dwindirectml.directml.dll}.
     */
    private String directmlDllOverride;
    private long requestTimeoutMillis = 30_000L;
    /**
     * Timeout for the {@code summarize} method in milliseconds.
     * Phi-3 inference can take 60–180 s on CPU and 30–90 s on DirectML
     * depending on input length and hardware. The default of 300 000 ms
     * (5 minutes) gives enough headroom for first-run JIT warm-up and
     * long inputs. Set to 0 to fall back to {@link #requestTimeoutMillis}.
     */
    private long summarizeTimeoutMillis = 300_000L;
    /**
     * Working directory for the sidecar process; {@code null} = inherit.
     */
    private String workingDirectory;
    /**
     * Optional additional JVM args (free-form, e.g. {@code -Xmx2g}).
     */
    private String extraJvmArgs;

    /**
     * Filesystem path to the Phi-3 model directory, forwarded as
     * {@code -Dphi3.modelDir=<value>}. When blank, the sidecar resolves
     * the conventional location
     * {@code model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128}.
     */
    private String phi3ModelDirectory;

    /**
     * Backend selector for the Phi-3 summarizer, forwarded as
     * {@code -Dphi3.backend=<value>}. Accepted values:
     * {@code auto} (default), {@code directml}, {@code cpu}.
     */
    private String phi3Backend = "auto";

    /**
     * Maximum number of generated tokens for the Phi-3 summarizer,
     * forwarded as {@code -Dphi3.maxTokens=<value>}. 0 means "use sidecar
     * default" (512).
     */
    private int phi3MaxTokens = 0;

    // ── getters / setters ───────────────────────────────────────────────

    public String getJavaExecutable() {
        return javaExecutable;
    }

    public void setJavaExecutable(String v) {
        this.javaExecutable = v;
    }

    public String getSidecarJarPath() {
        return sidecarJarPath;
    }

    public void setSidecarJarPath(String v) {
        this.sidecarJarPath = v;
    }

    public String getModelDirectory() {
        return modelDirectory;
    }

    public void setModelDirectory(String v) {
        this.modelDirectory = v;
    }

    public String getEmbedBackend() {
        return embedBackend;
    }

    public void setEmbedBackend(String v) {
        this.embedBackend = v;
    }

    public String getEmbedModel() {
        return embedModel;
    }

    public void setEmbedModel(String v) {
        this.embedModel = v;
    }

    public String getE5Variant() {
        return e5Variant;
    }

    public void setE5Variant(String v) {
        this.e5Variant = v;
    }

    public String getE5ModelDirectory() {
        return e5ModelDirectory;
    }

    public void setE5ModelDirectory(String v) {
        this.e5ModelDirectory = v;
    }

    public String getRerankBackend() {
        return rerankBackend;
    }

    public void setRerankBackend(String v) {
        this.rerankBackend = v;
    }

    public String getRerankModelDirectory() {
        return rerankModelDirectory;
    }

    public void setRerankModelDirectory(String v) {
        this.rerankModelDirectory = v;
    }

    public boolean isDirectmlDebug() {
        return directmlDebug;
    }

    public void setDirectmlDebug(boolean v) {
        this.directmlDebug = v;
    }

    public String getDirectmlDllOverride() {
        return directmlDllOverride;
    }

    public void setDirectmlDllOverride(String v) {
        this.directmlDllOverride = v;
    }

    public long getRequestTimeoutMillis() {
        return requestTimeoutMillis;
    }

    public void setRequestTimeoutMillis(long v) {
        this.requestTimeoutMillis = v;
    }

    public long getSummarizeTimeoutMillis() {
        return summarizeTimeoutMillis > 0 ? summarizeTimeoutMillis : requestTimeoutMillis;
    }

    public void setSummarizeTimeoutMillis(long v) {
        this.summarizeTimeoutMillis = v;
    }

    public String getWorkingDirectory() {
        return workingDirectory;
    }

    public void setWorkingDirectory(String v) {
        this.workingDirectory = v;
    }

    public String getExtraJvmArgs() {
        return extraJvmArgs;
    }

    public void setExtraJvmArgs(String v) {
        this.extraJvmArgs = v;
    }

    public String getPhi3ModelDirectory() {
        return phi3ModelDirectory;
    }

    public void setPhi3ModelDirectory(String v) {
        this.phi3ModelDirectory = v;
    }

    public String getPhi3Backend() {
        return phi3Backend;
    }

    public void setPhi3Backend(String v) {
        this.phi3Backend = v;
    }

    public int getPhi3MaxTokens() {
        return phi3MaxTokens;
    }

    public void setPhi3MaxTokens(int v) {
        this.phi3MaxTokens = v;
    }
}
