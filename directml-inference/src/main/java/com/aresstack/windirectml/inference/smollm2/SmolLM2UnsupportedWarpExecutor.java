package com.aresstack.windirectml.inference.smollm2;

import java.util.Objects;

/**
 * Placeholder executor that fails explicitly until native/WARP SmolLM2 kernels exist.
 */
public final class SmolLM2UnsupportedWarpExecutor implements SmolLM2WarpExecutor {

    private final String reason;

    public SmolLM2UnsupportedWarpExecutor() {
        this(SmolLM2WarpExecutionStatus.NATIVE_EXECUTOR_MISSING);
    }

    public SmolLM2UnsupportedWarpExecutor(String reason) {
        this.reason = reason == null || reason.isBlank()
                ? SmolLM2WarpExecutionStatus.NATIVE_EXECUTOR_MISSING
                : reason;
    }

    @Override
    public SmolLM2WarpExecutionStatus inspect(SmolLM2WarpRuntimePlan plan) {
        return SmolLM2WarpExecutionStatus.unsupported(plan, reason);
    }

    @Override
    public SmolLM2TokenRuntimeResult generate(SmolLM2Weights weights,
                                              SmolLM2WarpRuntimePlan plan,
                                              SmolLM2TokenRuntimeRequest request) {
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(request, "request");
        throw new SmolLM2RuntimeUnsupportedException(reason);
    }
}
