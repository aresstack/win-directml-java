package com.aresstack.windirectml.inference.smollm2;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Default session adapter for legacy {@link SmolLM2WarpExecutor} implementations.
 */
final class SmolLM2ExecutorBackedWarpSession implements SmolLM2WarpSession {

    private final SmolLM2WarpExecutor executor;
    private final SmolLM2Weights weights;
    private final SmolLM2WarpRuntimePlan plan;
    private final SmolLM2WarpUploadManifest uploadManifest;
    private final SmolLM2WarpExecutionStatus status;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    SmolLM2ExecutorBackedWarpSession(SmolLM2WarpExecutor executor,
                                     SmolLM2Weights weights,
                                     SmolLM2WarpRuntimePlan plan) {
        this.executor = Objects.requireNonNull(executor, "executor");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.uploadManifest = SmolLM2WarpUploadManifest.fromPlan(plan);
        this.status = Objects.requireNonNull(executor.inspect(plan), "status");
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
    public SmolLM2TokenRuntimeResult generateTokenIds(SmolLM2TokenRuntimeRequest request,
                                                      java.util.function.IntConsumer generatedTokenConsumer) {
        Objects.requireNonNull(request, "request");
        ensureOpen();
        if (!status.executable()) {
            throw new SmolLM2RuntimeUnsupportedException(status.reason());
        }
        // Legacy executor-backed sessions produce the full token batch in one call and cannot stream
        // incrementally, so {@code generatedTokenConsumer} is intentionally ignored here.
        return executor.generate(weights, plan, request);
    }

    @Override
    public boolean closed() {
        return closed.get();
    }

    @Override
    public void close() {
        closed.set(true);
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("SmolLM2 WARP session is closed");
        }
    }
}
