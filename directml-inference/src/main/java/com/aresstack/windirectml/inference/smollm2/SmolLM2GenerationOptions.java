package com.aresstack.windirectml.inference.smollm2;

/**
 * Optional generation parameters for the future SmolLM2 runtime.
 */
public record SmolLM2GenerationOptions(Double temperature, Integer topK, Double topP) {
    public static SmolLM2GenerationOptions greedy() {
        return new SmolLM2GenerationOptions(0.0d, null, null);
    }
}
