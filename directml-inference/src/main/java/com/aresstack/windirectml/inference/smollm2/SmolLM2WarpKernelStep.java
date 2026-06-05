package com.aresstack.windirectml.inference.smollm2;

import java.util.Objects;

/**
 * Planned kernel boundary for the future SmolLM2 WARP pipeline.
 */
public record SmolLM2WarpKernelStep(String name,
                                    int layerIndex,
                                    String inputBuffer,
                                    String outputBuffer) {

    public SmolLM2WarpKernelStep {
        name = Objects.requireNonNull(name, "name");
        inputBuffer = inputBuffer == null ? "" : inputBuffer;
        outputBuffer = outputBuffer == null ? "" : outputBuffer;
    }

    public boolean layerBound() {
        return layerIndex >= 0;
    }
}
