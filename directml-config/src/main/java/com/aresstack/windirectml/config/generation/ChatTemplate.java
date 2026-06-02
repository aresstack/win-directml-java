package com.aresstack.windirectml.config.generation;

/**
 * Describes the chat/instruction template format expected by a generation model.
 *
 * <p>Different model families use different prompt wrapping conventions
 * (e.g. Phi-3 uses {@code <|user|>...<|end|>}, Qwen uses
 * {@code <|im_start|>user...}). This enum classifies the known templates
 * so callers can format prompts correctly without hard-coding model names.
 *
 * <p>Java-8 compatible.
 */
public enum ChatTemplate {

    /**
     * Phi-3 instruction template ({@code <|user|>...<|end|><|assistant|>}).
     */
    PHI3,

    /**
     * Qwen/ChatML-style template ({@code <|im_start|>user\n...<|im_end|>}).
     */
    CHATML,

    /**
     * Raw prompt with no template wrapping.
     */
    RAW,

    /**
     * Template is not yet determined (planned models).
     */
    UNKNOWN;
}
