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
 *   <li><b>No hard-coded default system prompt.</b> The system turn carries only
 *       the caller's explicit override (or nothing). This removes the old
 *       "You are a helpful AI assistant" fallback that competed with the task.</li>
 *   <li><b>Task instruction lives in the <em>user</em> turn</b>, prepended to the
 *       raw text. Tiny instruction models (SmolLM2 135M/360M) barely attend to a
 *       system-turn directive and tend to echo or ignore it; placing the
 *       instruction immediately before the text — as real instruct usage does —
 *       makes them actually follow the task.</li>
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
        String system = resolveSystemTurn(input.systemOverride());
        String userTurn = composeUserTurn(task, input.systemOverride(), input.userText());
        ChatConversation conversation = ChatConversation.builder()
                .system(system)
                .user(userTurn)
                .build();
        return render(conversation);
    }

    /**
     * The system turn carries only an explicit caller override. Task instructions
     * go into the user turn (see {@link #composeUserTurn}).
     */
    private static String resolveSystemTurn(String systemOverride) {
        if (systemOverride != null && !systemOverride.isBlank()) {
            return systemOverride.trim();
        }
        return "";
    }

    /**
     * Place the raw user text first and the task instruction LAST, immediately
     * before the assistant turn. Tiny instruction models are strongly recency-biased:
     * the directive closest to the generation point is the one they actually act on.
     * When a system override is present it owns the directive, so the user turn stays
     * the raw text.
     */
    private String composeUserTurn(PromptTask task, String systemOverride, String userText) {
        String text = userText == null ? "" : userText;
        if (systemOverride != null && !systemOverride.isBlank()) {
            return text;
        }
        String instruction = instructions.instructionFor(task);
        if (instruction.isBlank()) {
            return text;
        }
        return text + "\n\n" + instruction;
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
