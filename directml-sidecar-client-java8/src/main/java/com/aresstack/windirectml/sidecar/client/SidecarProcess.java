package com.aresstack.windirectml.sidecar.client;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Manages the lifecycle of a single Java-21 DirectML sidecar process.
 *
 * <p>Owns:
 * <ul>
 *   <li>the underlying {@link Process} created via {@link ProcessBuilder};</li>
 *   <li>line-oriented readers for stdout (JSON-RPC) and stderr (logs);</li>
 *   <li>a line-oriented writer for stdin (outgoing JSON-RPC).</li>
 * </ul>
 *
 * <p>The class is intentionally pure Java-8. It does <b>not</b> parse JSON;
 * the caller (typically {@link SidecarClient}) is responsible for the
 * JSON-RPC framing on top of the line streams.
 */
public class SidecarProcess {

    /**
     * UTF-8 explicit; the sidecar emits UTF-8 regardless of platform default.
     */
    private static final Charset UTF8 = Charset.forName("UTF-8");

    private final SidecarClientConfig config;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final StringBuilder stderrBuffer = new StringBuilder();
    private final Object stderrLock = new Object();

    private Process process;
    private BufferedReader stdout;
    private BufferedWriter stdin;
    private Thread stderrPump;
    private List<String> commandLine = Collections.emptyList();

    public SidecarProcess(SidecarClientConfig config) {
        if (config == null) throw new IllegalArgumentException("config must not be null");
        this.config = config;
    }

    /**
     * Spawn the sidecar process. Returns once the child JVM is reported
     * "running" by the OS – it does not wait for the sidecar to be ready.
     */
    public synchronized void start() throws SidecarException {
        if (running.get()) {
            throw new SidecarException("Sidecar process already running");
        }
        if (isBlank(config.getSidecarJarPath())) {
            throw new SidecarException("SidecarClientConfig.sidecarJarPath must be set");
        }
        commandLine = buildCommandLine(config);
        ProcessBuilder pb = new ProcessBuilder(commandLine);
        if (!isBlank(config.getWorkingDirectory())) {
            File dir = new File(config.getWorkingDirectory());
            if (dir.isDirectory()) pb.directory(dir);
        }
        // Do NOT merge stderr into stdout: stdout is JSON-RPC-only.
        pb.redirectErrorStream(false);
        try {
            process = pb.start();
        } catch (IOException e) {
            throw new SidecarException("Failed to start sidecar process: " + e.getMessage(), e);
        }

        stdout = new BufferedReader(new InputStreamReader(process.getInputStream(), UTF8));
        stdin = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), UTF8));

        // Background thread: pump stderr into an in-memory buffer that
        // the UI / client can poll. Never blocks on stdout.
        final BufferedReader stderrReader = new BufferedReader(
                new InputStreamReader(process.getErrorStream(), UTF8));
        stderrPump = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    String line;
                    while ((line = stderrReader.readLine()) != null) {
                        synchronized (stderrLock) {
                            stderrBuffer.append(line).append('\n');
                            // Trim to a bounded window so we do not grow without limit.
                            if (stderrBuffer.length() > 1_000_000) {
                                stderrBuffer.delete(0, stderrBuffer.length() - 800_000);
                            }
                        }
                    }
                } catch (IOException ignored) {
                    // Stream closed when the process exits – normal shutdown.
                } finally {
                    try {
                        stderrReader.close();
                    } catch (IOException ignored2) { /* ignore */ }
                }
            }
        }, "sidecar-stderr-pump");
        stderrPump.setDaemon(true);
        stderrPump.start();
        running.set(true);
    }

    /**
     * Read the next JSON-RPC line from the sidecar's stdout. Blocks until
     * a line is available or the stream is closed; returns {@code null} on
     * end-of-stream.
     */
    public String readLine() throws IOException {
        BufferedReader r = stdout;
        return (r == null) ? null : r.readLine();
    }

    /**
     * Write a single JSON-RPC line to the sidecar's stdin, terminated by
     * a newline character. The line must not itself contain newlines.
     */
    public synchronized void writeLine(String line) throws IOException {
        BufferedWriter w = stdin;
        if (w == null) throw new IOException("Sidecar stdin not available");
        w.write(line);
        w.write('\n');
        w.flush();
    }

    public boolean isRunning() {
        if (!running.get()) return false;
        Process p = process;
        return p != null && p.isAlive();
    }

    /**
     * Process exit code or {@code -1} if still running.
     */
    public int exitValue() {
        Process p = process;
        if (p == null || p.isAlive()) return -1;
        try {
            return p.exitValue();
        } catch (IllegalThreadStateException e) {
            return -1;
        }
    }

    /**
     * Best-effort graceful stop: close stdin so the sidecar's main loop
     * exits, then wait up to {@code timeoutMillis} for the process to
     * terminate. After the timeout, force-kill.
     */
    public synchronized void stop(long timeoutMillis) {
        if (!running.compareAndSet(true, false)) return;
        // Close stdin first so the sidecar's main loop sees EOF.
        try {
            if (stdin != null) stdin.close();
        } catch (IOException ignored) { /* ignore */ }
        try {
            if (stdout != null) stdout.close();
        } catch (IOException ignored) { /* ignore */ }
        Process p = process;
        if (p == null) return;
        try {
            if (!p.waitFor(timeoutMillis, TimeUnit.MILLISECONDS)) {
                p.destroyForcibly();
                p.waitFor(2000, TimeUnit.MILLISECONDS);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            p.destroyForcibly();
        }
        if (stderrPump != null) {
            try {
                stderrPump.join(500);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
        }
    }

    public String getStderrSnapshot() {
        synchronized (stderrLock) {
            return stderrBuffer.toString();
        }
    }

    public List<String> getCommandLine() {
        return commandLine;
    }

    // ── command-line composition ────────────────────────────────────────

    /**
     * Build the {@code java …} command line that would be used to spawn
     * the sidecar with the given configuration. Useful for tests and for
     * UI preview before actually starting the process.
     */
    public static List<String> buildCommandLine(SidecarClientConfig cfg) {
        List<String> cmd = new ArrayList<String>();
        cmd.add(isBlank(cfg.getJavaExecutable()) ? "java" : cfg.getJavaExecutable());
        cmd.add("--enable-preview");
        cmd.add("--enable-native-access=ALL-UNNAMED");
        if (!isBlank(cfg.getExtraJvmArgs())) {
            String[] parts = cfg.getExtraJvmArgs().trim().split("\\s+");
            cmd.addAll(Arrays.asList(parts));
        }
        if (!isBlank(cfg.getEmbedBackend())) {
            cmd.add("-Dembed.backend=" + cfg.getEmbedBackend());
        }
        cmd.add("-Dwindirectml.debug=" + Boolean.toString(cfg.isDirectmlDebug()));
        if (!isBlank(cfg.getDirectmlDllOverride())) {
            cmd.add("-Dwindirectml.directml.dll=" + cfg.getDirectmlDllOverride());
        }
        if (!isBlank(cfg.getModelDirectory())) {
            cmd.add("-Dminilm.modelDir=" + cfg.getModelDirectory());
        }
        cmd.add("-jar");
        cmd.add(cfg.getSidecarJarPath());
        return cmd;
    }

    private static boolean isBlank(String s) {
        if (s == null) return true;
        for (int i = 0; i < s.length(); i++) {
            if (!Character.isWhitespace(s.charAt(i))) return false;
        }
        return true;
    }
}

