package com.aresstack.windirectml.sidecar.workbench;

import com.aresstack.windirectml.sidecar.client.EmbeddingResult;
import com.aresstack.windirectml.sidecar.client.HealthResult;
import com.aresstack.windirectml.sidecar.client.RerankResult;
import com.aresstack.windirectml.sidecar.client.SidecarClient;
import com.aresstack.windirectml.sidecar.client.SidecarClientConfig;
import com.aresstack.windirectml.sidecar.client.SidecarException;
import com.aresstack.windirectml.sidecar.client.SummaryResult;

/**
 * Owns the {@link SidecarClient} lifecycle for the workbench.
 *
 * <p>All blocking sidecar calls happen synchronously here. The UI must
 * dispatch them via {@code SwingWorker} so the EDT never blocks.
 */
public final class WorkbenchModel {

    private final SidecarClientConfig config = new SidecarClientConfig();
    private SidecarClient client;

    /**
     * Snapshot of the last stop, captured before {@link #client} is nulled
     * out so the UI can render a sensible message instead of {@code -1}.
     */
    private StopInfo lastStopInfo = StopInfo.never();

    /** Immutable snapshot of a sidecar stop. */
    public static final class StopInfo {
        private final boolean everStopped;
        private final boolean forced;
        private final Integer exitCode; // null = unknown

        private StopInfo(boolean everStopped, boolean forced, Integer exitCode) {
            this.everStopped = everStopped;
            this.forced = forced;
            this.exitCode = exitCode;
        }

        static StopInfo never() {
            return new StopInfo(false, false, null);
        }

        public boolean everStopped() { return everStopped; }
        public boolean forced() { return forced; }
        public Integer exitCode() { return exitCode; }

        /** Human-readable one-liner; never contains the misleading "exit=-1". */
        public String describe() {
            if (!everStopped) return "Sidecar stopped";
            String kind = forced ? "forced" : "clean";
            if (exitCode == null) return "Sidecar stopped (" + kind + ", exit code unavailable)";
            return "Sidecar stopped (" + kind + ", exit=" + exitCode + ")";
        }
    }

    public SidecarClientConfig getConfig() {
        return config;
    }

    public synchronized boolean isRunning() {
        return client != null && client.isRunning();
    }

    public synchronized void startSidecar() throws SidecarException {
        if (client != null && client.isRunning()) {
            throw new SidecarException("Sidecar already running");
        }
        client = new SidecarClient(config);
        client.start();
    }

    public synchronized void stopSidecar() {
        if (client != null) {
            client.shutdown();
            Integer code = client.lastExitCode();
            boolean forced = client.lastStopForced();
            lastStopInfo = new StopInfo(true, forced, code);
            client = null;
        }
    }

    public synchronized StopInfo lastStopInfo() {
        return lastStopInfo;
    }

    public synchronized void restartSidecar() throws SidecarException {
        stopSidecar();
        startSidecar();
    }

    public synchronized HealthResult health() throws SidecarException {
        if (client == null) throw new SidecarException("Sidecar is not running");
        return client.health();
    }

    public synchronized EmbeddingResult embed(String text) throws SidecarException {
        if (client == null) throw new SidecarException("Sidecar is not running");
        return client.embed(text);
    }

    public synchronized EmbeddingResult embed(String text, boolean normalize, String prefix)
            throws SidecarException {
        if (client == null) throw new SidecarException("Sidecar is not running");
        return client.embed(text, normalize, prefix);
    }

    public synchronized SummaryResult summarize(String text, int maxTokens) throws SidecarException {
        if (client == null) throw new SidecarException("Sidecar is not running");
        return client.summarize(text, maxTokens);
    }

    public synchronized RerankResult rerank(String query, java.util.List<String> documents, int topN)
            throws SidecarException {
        if (client == null) throw new SidecarException("Sidecar is not running");
        return client.rerank(query, documents, topN);
    }

    public synchronized String getLastRawRequest() {
        return client != null ? client.getLastRawRequest() : null;
    }

    public synchronized String getLastRawResponse() {
        return client != null ? client.getLastRawResponse() : null;
    }

    public synchronized String getStderrSnapshot() {
        return client != null ? client.getStderrSnapshot() : "";
    }

    public synchronized int exitValue() {
        return client != null ? client.exitValue() : -1;
    }

    public synchronized java.util.List<String> getCommandLine() {
        return client != null
                ? client.getCommandLine()
                : com.aresstack.windirectml.sidecar.client.SidecarProcess.buildCommandLine(config);
    }
}

