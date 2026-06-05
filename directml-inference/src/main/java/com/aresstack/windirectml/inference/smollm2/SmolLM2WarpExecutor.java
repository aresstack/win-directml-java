package com.aresstack.windirectml.inference.smollm2;

/**
 * Strategy boundary for the future native/WARP SmolLM2 executor.
 */
public interface SmolLM2WarpExecutor {

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
