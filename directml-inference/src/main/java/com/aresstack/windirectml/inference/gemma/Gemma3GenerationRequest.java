package com.aresstack.windirectml.inference.gemma;

import java.util.Objects;

/**
 * A native WARP generation request (GEMMA-WARP-10b): the (already tokenized) prompt and the maximum
 * number of new visible tokens to produce.
 */
public record Gemma3GenerationRequest(int[] promptIds, int maxNewTokens) {

    public Gemma3GenerationRequest {
        Objects.requireNonNull(promptIds, "promptIds");
        if (promptIds.length == 0) {
            throw new IllegalArgumentException("promptIds must not be empty");
        }
        if (maxNewTokens <= 0) {
            throw new IllegalArgumentException("maxNewTokens must be positive: " + maxNewTokens);
        }
    }
}
