package com.aresstack.windirectml.inference.smollm2;

import java.util.List;
import java.util.Objects;

/**
 * Static WARP preparation result that can be inspected before native resources exist.
 */
public record SmolLM2WarpRuntimePlan(SmolLM2Config config,
                                     int sequenceLength,
                                     SmolLM2WarpBufferPlan bufferPlan,
                                     SmolLM2WarpKernelPlan kernelPlan,
                                     List<String> warnings) {

    public SmolLM2WarpRuntimePlan {
        config = Objects.requireNonNull(config, "config");
        if (sequenceLength <= 0) {
            throw new IllegalArgumentException("sequenceLength must be positive");
        }
        bufferPlan = Objects.requireNonNull(bufferPlan, "bufferPlan");
        kernelPlan = Objects.requireNonNull(kernelPlan, "kernelPlan");
        warnings = warnings == null ? List.of() : List.copyOf(warnings);
    }

    public boolean readyForNativeAllocation() {
        return warnings.isEmpty();
    }
}
