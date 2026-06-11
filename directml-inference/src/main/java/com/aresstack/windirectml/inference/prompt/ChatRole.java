package com.aresstack.windirectml.inference.prompt;

/**
 * Role of a single message inside a model-neutral {@link ChatConversation}.
 *
 * <p>This enum is intentionally free of any wire-format detail. The mapping from
 * a role to concrete control tokens (ChatML {@code <|im_start|>}, Phi-3
 * {@code <|system|>}, T5 task prefixes, ...) is the responsibility of a
 * model-family specific {@link PromptStrategy}.</p>
 */
public enum ChatRole {
    SYSTEM,
    USER,
    ASSISTANT
}
