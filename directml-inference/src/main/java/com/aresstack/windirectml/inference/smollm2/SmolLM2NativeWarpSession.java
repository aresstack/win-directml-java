package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.windows.WindowsBindings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Native DirectML/WARP session for SmolLM2.
 *
 * <p>Owns the {@link WindowsBindings} (initialised on the {@code warp} backend) and a {@link SmolLM2WarpForwardPass}
 * whose projection weights are uploaded once and reused across every {@code generateTokenIds} call. If the WARP
 * device cannot be initialised or the weights cannot be uploaded, the session reports a non-executable status so the
 * {@code AUTO} runtime mode transparently falls back to the CPU reference path.</p>
 */
final class SmolLM2NativeWarpSession implements SmolLM2WarpSession {

    private static final Logger log = LoggerFactory.getLogger(SmolLM2NativeWarpSession.class);

    private final SmolLM2Weights weights;
    private final SmolLM2WarpRuntimePlan plan;
    private final SmolLM2WarpUploadManifest uploadManifest;
    private final SmolLM2WarpExecutionStatus status;
    private final String backend;
    private final boolean supported;

    private final Object engineLock = new Object();
    private WindowsBindings windowsBindings;        // built lazily on first generate
    private SmolLM2WarpForwardPass forwardPass;      // built lazily on first generate
    private SmolLM2WarpGenerationLoop generationLoop; // built lazily on first generate
    private boolean engineBuilt;

    private final AtomicBoolean closed = new AtomicBoolean(false);

    SmolLM2NativeWarpSession(SmolLM2Weights weights, SmolLM2WarpRuntimePlan plan, String backend) {
        this.weights = Objects.requireNonNull(weights, "weights");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.uploadManifest = SmolLM2WarpUploadManifest.fromPlan(plan);
        this.backend = backend == null || backend.isBlank() ? "warp" : backend.trim();
        this.supported = WindowsBindings.isSupported();

        List<String> warnings = new ArrayList<>(plan.warnings());
        if (supported) {
            // Readiness is reported optimistically without uploading any weights yet. The projection
            // weights are uploaded lazily on the first generate call so the readiness probe stays cheap
            // (the workbench prepares a runtime once just to inspect readiness, then opens the real one).
            this.status = new SmolLM2WarpExecutionStatus(true, "warp",
                    "SmolLM2 native WARP executor available (backend=" + this.backend + ").", warnings);
        } else {
            String reason = "SmolLM2 native WARP executor unavailable: Windows D3D12/DirectML is not supported on this host.";
            warnings.add(reason);
            this.status = new SmolLM2WarpExecutionStatus(false, "warp", reason, warnings);
        }
    }

    @Override
    public SmolLM2WarpRuntimePlan plan() {
        return plan;
    }

    @Override
    public SmolLM2WarpUploadManifest uploadManifest() {
        return uploadManifest;
    }

    @Override
    public SmolLM2WarpExecutionStatus status() {
        return status;
    }

    @Override
    public SmolLM2TokenRuntimeResult generateTokenIds(SmolLM2TokenRuntimeRequest request) {
        Objects.requireNonNull(request, "request");
        ensureOpen();
        if (!supported) {
            throw new SmolLM2RuntimeUnsupportedException(status.reason());
        }
        return ensureEngine().generate(request, null);
    }

    private SmolLM2WarpGenerationLoop ensureEngine() {
        synchronized (engineLock) {
            ensureOpen();
            if (engineBuilt) {
                return generationLoop;
            }
            WindowsBindings bindings = null;
            SmolLM2WarpForwardPass builtForwardPass = null;
            try {
                bindings = new WindowsBindings();
                bindings.init(backend);
                long t0 = System.currentTimeMillis();
                builtForwardPass = new SmolLM2WarpForwardPass(bindings, weights);
                log.info("SmolLM2 native WARP engine ready (backend={}, {} layers uploaded in {} ms)",
                        backend, weights.layers().size(), System.currentTimeMillis() - t0);
                this.windowsBindings = bindings;
                this.forwardPass = builtForwardPass;
                this.generationLoop = new SmolLM2WarpGenerationLoop(builtForwardPass);
                this.engineBuilt = true;
                return generationLoop;
            } catch (Exception e) {
                if (builtForwardPass != null) {
                    try {
                        builtForwardPass.close();
                    } catch (RuntimeException ignored) {
                        // best-effort cleanup
                    }
                }
                if (bindings != null) {
                    try {
                        bindings.close();
                    } catch (RuntimeException ignored) {
                        // best-effort cleanup
                    }
                }
                throw new SmolLM2RuntimeUnsupportedException(
                        "SmolLM2 native WARP engine failed to initialise (backend=" + backend + "): "
                                + safeMessage(e), e);
            }
        }
    }

    @Override
    public boolean closed() {
        return closed.get();
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            synchronized (engineLock) {
                if (forwardPass != null) {
                    try {
                        forwardPass.close();
                    } catch (RuntimeException e) {
                        log.warn("Error closing SmolLM2 WARP forward pass: {}", e.getMessage());
                    }
                    forwardPass = null;
                }
                if (windowsBindings != null) {
                    try {
                        windowsBindings.close();
                    } catch (RuntimeException e) {
                        log.warn("Error closing SmolLM2 WARP WindowsBindings: {}", e.getMessage());
                    }
                    windowsBindings = null;
                }
                generationLoop = null;
            }
        }
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("SmolLM2 native WARP session is closed");
        }
    }

    private static String safeMessage(Throwable e) {
        return e.getMessage() == null || e.getMessage().isBlank()
                ? e.getClass().getSimpleName()
                : e.getMessage();
    }
}
