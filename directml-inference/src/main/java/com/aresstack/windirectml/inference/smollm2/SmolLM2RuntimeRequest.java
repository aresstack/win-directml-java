package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.prompt.PromptInput;

/**
 * Request contract for SmolLM2 text generation.
 *
 * <p>The prompt is supplied as a model-neutral {@link PromptInput} (task + raw
 * text + optional system override). The owning {@link SmolLM2Runtime} renders it
 * into the ChatML wire format through its {@code PromptStrategy} – the Workbench
 * never builds prompt strings itself.</p>
 */
public record SmolLM2RuntimeRequest(PromptInput prompt, int maxNewTokens, SmolLM2GenerationOptions options) {
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
