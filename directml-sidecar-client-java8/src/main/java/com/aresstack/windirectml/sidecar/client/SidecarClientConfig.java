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
    private boolean directmlDebug = false;
    /**
     * Optional override DLL path forwarded as {@code -Dwindirectml.directml.dll}.
     */
    private String directmlDllOverride;
    private long requestTimeoutMillis = 30_000L;
    /**
     * Working directory for the sidecar process; {@code null} = inherit.
     */
    private String workingDirectory;
    /**
     * Optional additional JVM args (free-form, e.g. {@code -Xmx2g}).
     */
    private String extraJvmArgs;

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
}

