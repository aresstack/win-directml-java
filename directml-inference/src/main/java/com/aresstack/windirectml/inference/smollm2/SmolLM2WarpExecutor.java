package com.aresstack.windirectml.inference.smollm2;

/**
 * Strategy boundary for the native/WARP SmolLM2 executor. The production implementation is
 * {@link SmolLM2NativeWarpExecutor}; probe/unsupported implementations exist for diagnostics and AUTO fallback tests.
 */
public interface SmolLM2WarpExecutor {

    /**
     * Open a prepared execution session for the runtime plan.
     */
    default SmolLM2WarpSession openSession(SmolLM2Weights weights, SmolLM2WarpRuntimePlan plan) {
        return new SmolLM2ExecutorBackedWarpSession(this, weights, plan);
    }

    /**
     * Report whether this executor can run the prepared plan on the current machine.
     */
    SmolLM2WarpExecutionStatus inspect(SmolLM2WarpRuntimePlan plan);

    /**
     * Execute token generation for a prepared runtime plan.
     */
    SmolLM2TokenRuntimeResult generate(SmolLM2Weights weights,
                                       SmolLM2WarpRuntimePlan plan,
                                       SmolLM2TokenRuntimeRequest request);
}
