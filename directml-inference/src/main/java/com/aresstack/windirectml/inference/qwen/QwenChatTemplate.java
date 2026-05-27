package com.aresstack.windirectml.inference.qwen;

import java.util.List;

/**
 * ChatML-style prompt template for Qwen2.5-Coder-Instruct models.
 *
 * <h2>Format</h2>
 * Qwen uses the ChatML convention:
 * <pre>
 * &lt;|im_start|&gt;system
 * {system_message}&lt;|im_end|&gt;
 * &lt;|im_start|&gt;user
 * {user_message}&lt;|im_end|&gt;
 * &lt;|im_start|&gt;assistant
 * </pre>
 *
 * <h2>Differences from Phi-3 chat template</h2>
 * <ul>
 *   <li>Phi-3 uses dedicated role tokens ({@code <|system|>}, {@code <|user|>},
 *       {@code <|assistant|>}) followed by a newline, then content, then {@code <|end|>}.</li>
 *   <li>Qwen/ChatML uses a generic {@code <|im_start|>} marker followed by the role
 *       name as plain text on the same line, a newline, then content, then
 *       {@code <|im_end|>}.</li>
 *   <li>Phi-3 places the role token on its own line; ChatML keeps role on the
 *       same line as {@code <|im_start|>}.</li>
 * </ul>
 *
 * <p><b>Runtime status:</b> Qwen model generation is <em>planned</em>.
 * This template supports offline prompt preparation and testing only.
 */
public final class QwenChatTemplate {

    /**
     * Qwen official default system prompt used when no system message is provided.
     */
    public static final String DEFAULT_SYSTEM_PROMPT =
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

    private QwenChatTemplate() {} // utility class

    /**
     * A single message in a multi-turn conversation.
     *
     * @param role    one of {@code "system"}, {@code "user"}, or {@code "assistant"}
     * @param content the message text
     */
    public record ChatMessage(String role, String content) {
        public static ChatMessage system(String content) { return new ChatMessage("system", content); }
        public static ChatMessage user(String content) { return new ChatMessage("user", content); }
        public static ChatMessage assistant(String content) { return new ChatMessage("assistant", content); }
    }

    /**
     * Format a single-turn chat prompt (system + user) using ChatML.
     *
     * <p>The output ends with {@code <|im_start|>assistant\n} to prompt
     * the model to begin generating.
     *
     * @param systemPrompt system instruction (may be {@code null} or empty)
     * @param userMessage  user message
     * @return formatted prompt string ready for encoding
     */
    public static String formatChat(String systemPrompt, String userMessage) {
        return formatChat(systemPrompt, userMessage, true);
    }

    /**
     * Format a single-turn chat prompt (system + user) using ChatML.
     *
     * @param systemPrompt system instruction (may be {@code null} or empty)
     * @param userMessage user message
     * @param useDefaultSystemPrompt if true, null/empty system prompts are replaced
     *                               with {@link #DEFAULT_SYSTEM_PROMPT}; if false,
     *                               the system turn is omitted
     * @return formatted prompt string ready for encoding
     */
    public static String formatChat(String systemPrompt, String userMessage, boolean useDefaultSystemPrompt) {
        return formatMultiTurnChat(systemPrompt, List.of(ChatMessage.user(userMessage)), useDefaultSystemPrompt);
    }

    /**
     * Format a multi-turn chat prompt using ChatML conventions.
     *
     * <pre>
     * &lt;|im_start|&gt;system
     * {systemPrompt}&lt;|im_end|&gt;
     * &lt;|im_start|&gt;user
     * {message1}&lt;|im_end|&gt;
     * &lt;|im_start|&gt;assistant
     * {reply1}&lt;|im_end|&gt;
     * &lt;|im_start|&gt;user
     * {message2}&lt;|im_end|&gt;
     * &lt;|im_start|&gt;assistant
     * </pre>
     *
     * @param systemPrompt optional system prompt (may be {@code null} or empty)
     * @param messages     ordered list of user/assistant messages; the last
     *                     message should typically be a user message
     * @return formatted prompt string ready for encoding
     */
    public static String formatMultiTurnChat(String systemPrompt, List<ChatMessage> messages) {
        return formatMultiTurnChat(systemPrompt, messages, true);
    }

    /**
     * Format a multi-turn chat prompt using ChatML conventions.
     *
     * @param systemPrompt optional system prompt (may be {@code null} or empty)
     * @param messages ordered list of user/assistant messages; the last
     *                 message should typically be a user message
     * @param useDefaultSystemPrompt if true, null/empty system prompts are replaced
     *                               with {@link #DEFAULT_SYSTEM_PROMPT}; if false,
     *                               the system turn is omitted
     * @return formatted prompt string ready for encoding
     */
    public static String formatMultiTurnChat(String systemPrompt, List<ChatMessage> messages,
                                             boolean useDefaultSystemPrompt) {
        StringBuilder sb = new StringBuilder();
        String effectiveSystemPrompt = systemPrompt;
        if (effectiveSystemPrompt == null || effectiveSystemPrompt.isEmpty()) {
            effectiveSystemPrompt = useDefaultSystemPrompt ? DEFAULT_SYSTEM_PROMPT : null;
        }

        if (effectiveSystemPrompt != null && !effectiveSystemPrompt.isEmpty()) {
            sb.append("<|im_start|>system\n")
              .append(effectiveSystemPrompt)
              .append("<|im_end|>\n");
        }

        for (ChatMessage msg : messages) {
            sb.append("<|im_start|>").append(msg.role()).append('\n')
              .append(msg.content())
              .append("<|im_end|>\n");
        }

        // Add generation prompt
        sb.append("<|im_start|>assistant\n");
        return sb.toString();
    }
}
