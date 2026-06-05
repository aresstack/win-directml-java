package com.aresstack.windirectml.inference.smollm2;

import java.util.Objects;

/**
 * Request contract for future SmolLM2 text generation.
 */
public record SmolLM2RuntimeRequest(String prompt, int maxNewTokens, SmolLM2GenerationOptions options) {
    public SmolLM2RuntimeRequest {
        if (prompt == null) {
            throw new IllegalArgumentException("prompt must not be null");
        }
        if (maxNewTokens <= 0) {
            throw new IllegalArgumentException("maxNewTokens must be > 0");
        }
        options = options == null ? SmolLM2GenerationOptions.greedy() : options;
    }
}
