package com.aresstack.windirectml.inference.smollm2;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Prepared SmolLM2 WARP runtime boundary.
 *
 * <p>This class does not pretend that native kernels already exist. It prepares the deterministic plan and delegates
 * execution to a {@link SmolLM2WarpExecutor}. The default executor reports that WARP execution is not implemented yet.</p>
 */
public final class SmolLM2WarpRuntime implements AutoCloseable {

    private final SmolLM2WarpSession session;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private SmolLM2WarpRuntime(SmolLM2WarpSession session) {
        this.session = Objects.requireNonNull(session, "session");
    }

    public static SmolLM2WarpRuntime prepare(SmolLM2RuntimePackage runtimePackage, int sequenceLength) {
        return prepare(runtimePackage, sequenceLength, SmolLM2WarpExecutorFactory.createDefaultExecutor());
    }

    public static SmolLM2WarpRuntime prepare(SmolLM2RuntimePackage runtimePackage,
                                             int sequenceLength,
                                             SmolLM2WarpExecutor executor) {
        Objects.requireNonNull(runtimePackage, "runtimePackage");
        Objects.requireNonNull(executor, "executor");
        if (!runtimePackage.executable()) {
            throw new SmolLM2RuntimeUnsupportedException(runtimePackage.runtimeLoadableReason());
        }
        SmolLM2Weights weights = runtimePackage.requireWeights();
        SmolLM2WarpRuntimePlan plan = new SmolLM2WarpRuntimePlanner().plan(weights, sequenceLength);
        SmolLM2WarpSession session = executor.openSession(weights, plan);
        return new SmolLM2WarpRuntime(session);
    }

    public SmolLM2TokenRuntimeResult generateTokenIds(SmolLM2TokenRuntimeRequest request) {
        Objects.requireNonNull(request, "request");
        ensureOpen();
        if (!session.status().executable()) {
            throw new SmolLM2RuntimeUnsupportedException(session.status().reason());
        }
        return session.generateTokenIds(request);
    }

    public SmolLM2WarpRuntimePlan plan() {
        return session.plan();
    }

    public SmolLM2WarpExecutionStatus status() {
        return session.status();
    }

    public SmolLM2WarpUploadManifest uploadManifest() {
        return session.uploadManifest();
    }

    public boolean executable() {
        return session.status().executable();
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("SmolLM2 WARP runtime is closed");
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            session.close();
        }
    }
}
