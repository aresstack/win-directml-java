package com.aresstack.windirectml.inference.smollm2;

import java.util.Objects;

/**
 * Placeholder executor that fails explicitly until native/WARP SmolLM2 kernels exist.
 */
public final class SmolLM2UnsupportedWarpExecutor implements SmolLM2WarpExecutor {

    @Override
    public SmolLM2WarpExecutionStatus inspect(SmolLM2WarpRuntimePlan plan) {
        return SmolLM2WarpExecutionStatus.unsupported(plan);
    }

    @Override
    public SmolLM2TokenRuntimeResult generate(SmolLM2Weights weights,
                                              SmolLM2WarpRuntimePlan plan,
                                              SmolLM2TokenRuntimeRequest request) {
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(request, "request");
        throw new SmolLM2RuntimeUnsupportedException(SmolLM2WarpExecutionStatus.NATIVE_EXECUTOR_MISSING);
    }
}
