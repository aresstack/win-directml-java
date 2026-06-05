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

    private final SmolLM2Weights weights;
    private final SmolLM2WarpRuntimePlan plan;
    private final SmolLM2WarpExecutor executor;
    private final SmolLM2WarpExecutionStatus status;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private SmolLM2WarpRuntime(SmolLM2Weights weights,
                               SmolLM2WarpRuntimePlan plan,
                               SmolLM2WarpExecutor executor,
                               SmolLM2WarpExecutionStatus status) {
        this.weights = Objects.requireNonNull(weights, "weights");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.executor = Objects.requireNonNull(executor, "executor");
        this.status = Objects.requireNonNull(status, "status");
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
        SmolLM2WarpExecutionStatus status = executor.inspect(plan);
        return new SmolLM2WarpRuntime(weights, plan, executor, status);
    }

    public SmolLM2TokenRuntimeResult generateTokenIds(SmolLM2TokenRuntimeRequest request) {
        Objects.requireNonNull(request, "request");
        ensureOpen();
        if (!status.executable()) {
            throw new SmolLM2RuntimeUnsupportedException(status.reason());
        }
        return executor.generate(weights, plan, request);
    }

    public SmolLM2WarpRuntimePlan plan() {
        return plan;
    }

    public SmolLM2WarpExecutionStatus status() {
        return status;
    }

    public boolean executable() {
        return status.executable();
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("SmolLM2 WARP runtime is closed");
        }
    }

    @Override
    public void close() {
        closed.set(true);
    }
}
