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
}

