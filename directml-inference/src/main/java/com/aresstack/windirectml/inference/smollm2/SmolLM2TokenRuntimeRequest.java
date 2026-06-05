package com.aresstack.windirectml.inference.smollm2;

import java.util.List;

/**
 * Token-level request for the SmolLM2 reference runtime.
 *
 * <p>This contract intentionally avoids tokenizer ownership. The public text API can be enabled later when a
 * production tokenizer is available.</p>
 */
public record SmolLM2TokenRuntimeRequest(List<Integer> inputTokenIds,
                                         int maxNewTokens,
                                         SmolLM2GenerationOptions options) {
    public SmolLM2TokenRuntimeRequest {
        if (inputTokenIds == null || inputTokenIds.isEmpty()) {
            throw new IllegalArgumentException("inputTokenIds must not be empty");
        }
        inputTokenIds = List.copyOf(inputTokenIds);
        if (maxNewTokens <= 0) {
            throw new IllegalArgumentException("maxNewTokens must be > 0");
        }
        options = options == null ? SmolLM2GenerationOptions.greedy() : options;
    }
}
