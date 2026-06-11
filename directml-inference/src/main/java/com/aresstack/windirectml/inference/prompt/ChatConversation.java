package com.aresstack.windirectml.inference.prompt;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Model-neutral, ordered sequence of {@link ChatMessage}s.
 *
 * <p>This is the single hand-off type produced by the prompt pipeline and
 * consumed by a model-family specific {@link PromptStrategy}. It contains no
 * wire-format and no model-specific defaults.</p>
 */
public final class ChatConversation {

    private final List<ChatMessage> messages;

    private ChatConversation(List<ChatMessage> messages) {
        this.messages = Collections.unmodifiableList(new ArrayList<>(messages));
    }

    public static ChatConversation of(ChatMessage... messages) {
        return new ChatConversation(List.of(messages));
    }

    public static ChatConversation of(List<ChatMessage> messages) {
        return new ChatConversation(Objects.requireNonNull(messages, "messages"));
    }

    public static Builder builder() {
        return new Builder();
    }

    public List<ChatMessage> messages() {
        return messages;
    }

    /**
     * First system message, if any. Convenience for formatters that render a
     * dedicated system block separately from the turn list.
     */
    public Optional<ChatMessage> systemMessage() {
        return messages.stream().filter(m -> m.role() == ChatRole.SYSTEM).findFirst();
    }

    public boolean isEmpty() {
        return messages.isEmpty();
    }

    public static final class Builder {
        private final List<ChatMessage> messages = new ArrayList<>();

        public Builder system(String content) {
            return add(ChatMessage.system(content));
        }

        public Builder user(String content) {
            return add(ChatMessage.user(content));
        }

        public Builder assistant(String content) {
            return add(ChatMessage.assistant(content));
        }

        /** Adds a message unless it is a blank system/assistant turn (blank user turns are kept). */
        public Builder add(ChatMessage message) {
            Objects.requireNonNull(message, "message");
            if (message.role() != ChatRole.USER && message.isBlank()) {
                return this;
            }
            messages.add(message);
            return this;
        }

        public ChatConversation build() {
            return new ChatConversation(messages);
        }
    }
}
