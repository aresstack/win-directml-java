package com.aresstack.windirectml.sidecar.workbench;

import com.aresstack.windirectml.sidecar.client.EmbeddingResult;
import com.aresstack.windirectml.sidecar.client.HealthResult;
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
            client = null;
        }
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

    public synchronized SummaryResult summarize(String text, int maxTokens) throws SidecarException {
        if (client == null) throw new SidecarException("Sidecar is not running");
        return client.summarize(text, maxTokens);
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

