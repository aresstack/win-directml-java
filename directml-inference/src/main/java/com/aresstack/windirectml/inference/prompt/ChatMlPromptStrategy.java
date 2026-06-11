package com.aresstack.windirectml.inference.prompt;

import java.util.Objects;
import java.util.Set;

/**
 * {@link PromptStrategy} for ChatML-style instruction models (SmolLM2, Qwen).
 *
 * <p>Renders a model-neutral {@link PromptInput} into the ChatML wire format:</p>
 * <pre>
 * &lt;|im_start|&gt;system
 * {system}&lt;|im_end|&gt;
 * &lt;|im_start|&gt;user
 * {user}&lt;|im_end|&gt;
 * &lt;|im_start|&gt;assistant
 * </pre>
 *
 * <p>Key design points:</p>
 * <ul>
 *   <li><b>No hard-coded default system prompt.</b> The system turn is the
 *       caller's override, or the task instruction, or nothing. This removes the
 *       old "You are a helpful AI assistant" fallback that competed with the
 *       selected task.</li>
 *   <li>The task instruction goes into the <em>system</em> turn; the user turn
 *       stays the raw text.</li>
 *   <li>Already-rendered ChatML input is passed through untouched.</li>
 * </ul>
 */
public final class ChatMlPromptStrategy implements PromptStrategy {

    private static final String IM_START = "<|im_start|>";
    private static final String IM_END = "<|im_end|>";

    private final ChatTaskInstructions instructions;
    private final Set<PromptTask> supportedTasks;

    public ChatMlPromptStrategy(ChatTaskInstructions instructions, Set<PromptTask> supportedTasks) {
        this.instructions = Objects.requireNonNull(instructions, "instructions");
        this.supportedTasks = Set.copyOf(Objects.requireNonNull(supportedTasks, "supportedTasks"));
    }

    @Override
    public Set<PromptTask> supportedTasks() {
        return supportedTasks;
    }

    @Override
    public String renderPrompt(PromptInput input) {
        Objects.requireNonNull(input, "input");
        if (looksLikeRenderedChat(input.userText())) {
            return input.userText();
        }
        PromptTask task = supportedTasks.contains(input.task()) ? input.task() : PromptTask.NONE;
        String system = resolveSystemTurn(task, input.systemOverride());
        ChatConversation conversation = ChatConversation.builder()
                .system(system)
                .user(input.userText())
                .build();
        return render(conversation);
    }

    private String resolveSystemTurn(PromptTask task, String systemOverride) {
        if (systemOverride != null && !systemOverride.isBlank()) {
            return systemOverride.trim();
        }
        return instructions.instructionFor(task);
    }

    private static String render(ChatConversation conversation) {
        StringBuilder prompt = new StringBuilder();
        for (ChatMessage message : conversation.messages()) {
            prompt.append(IM_START)
                    .append(roleName(message.role()))
                    .append('\n')
                    .append(message.content())
                    .append(IM_END)
                    .append('\n');
        }
        prompt.append(IM_START).append("assistant\n");
        return prompt.toString();
    }

    private static String roleName(ChatRole role) {
        return switch (role) {
            case SYSTEM -> "system";
            case USER -> "user";
            case ASSISTANT -> "assistant";
        };
    }

    private static boolean looksLikeRenderedChat(String prompt) {
        return prompt != null && prompt.contains(IM_START) && prompt.contains(IM_END);
    }
}
