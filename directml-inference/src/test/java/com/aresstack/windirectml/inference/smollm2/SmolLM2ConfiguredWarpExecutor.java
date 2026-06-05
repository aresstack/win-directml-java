package com.aresstack.windirectml.inference.smollm2;

import java.util.List;

/**
 * Test executor loaded through the SmolLM2 WARP executor factory.
 */
public final class SmolLM2ConfiguredWarpExecutor implements SmolLM2WarpExecutor {

    @Override
    public SmolLM2WarpExecutionStatus inspect(SmolLM2WarpRuntimePlan plan) {
        return new SmolLM2WarpExecutionStatus(true, "warp", "configured test executor is available", List.of());
    }

    @Override
    public SmolLM2TokenRuntimeResult generate(SmolLM2Weights weights,
                                              SmolLM2WarpRuntimePlan plan,
                                              SmolLM2TokenRuntimeRequest request) {
        return new SmolLM2TokenRuntimeResult(
                request.inputTokenIds(),
                List.of(6),
                List.of(1, 2, 6),
                1,
                "length",
                request.maxNewTokens());
    }
}
