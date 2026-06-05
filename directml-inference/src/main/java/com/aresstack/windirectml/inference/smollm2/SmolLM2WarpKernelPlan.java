package com.aresstack.windirectml.inference.smollm2;

import java.util.List;
import java.util.Objects;

/**
 * Ordered kernel skeleton for future WARP execution.
 */
public record SmolLM2WarpKernelPlan(List<SmolLM2WarpKernelStep> steps) {

    public SmolLM2WarpKernelPlan {
        steps = List.copyOf(Objects.requireNonNull(steps, "steps"));
    }

    public int stepCount() {
        return steps.size();
    }
}
