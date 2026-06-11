package com.aresstack.windirectml.inference.prompt;

import java.util.Objects;

/**
 * A single, model-neutral message in a {@link ChatConversation}.
 *
 * <p>Carries only a {@link ChatRole} and the raw text. No control tokens, no
 * model-specific formatting. A {@link PromptStrategy} turns a sequence of these
 * into the exact wire string a concrete model family expects.</p>
 */
public record ChatMessage(ChatRole role, String content) {

    public ChatMessage {
        Objects.requireNonNull(role, "role");
        content = content == null ? "" : content;
    }

    public static ChatMessage system(String content) {
        return new ChatMessage(ChatRole.SYSTEM, content);
    }

    public static ChatMessage user(String content) {
        return new ChatMessage(ChatRole.USER, content);
    }

    public static ChatMessage assistant(String content) {
        return new ChatMessage(ChatRole.ASSISTANT, content);
    }

    public boolean isBlank() {
        return content.isBlank();
    }
}
