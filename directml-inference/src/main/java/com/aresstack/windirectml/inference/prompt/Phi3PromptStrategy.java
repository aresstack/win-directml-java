package com.aresstack.windirectml.inference.prompt;

import java.util.Objects;
import java.util.Set;

/**
 * {@link PromptStrategy} for Phi-3-mini-instruct models.
 *
 * <p>Renders the Phi-3 chat template:</p>
 * <pre>
 * &lt;|system|&gt;
 * {system}&lt;|end|&gt;
 * &lt;|user|&gt;
 * {user}&lt;|end|&gt;
 * &lt;|assistant|&gt;
 * </pre>
 *
 * <p>Like {@link ChatMlPromptStrategy}, there is no hard-coded default system
 * prompt: the system turn is the caller override, the task instruction, or
 * nothing.</p>
 */
public final class Phi3PromptStrategy implements PromptStrategy {

    private final ChatTaskInstructions instructions;
    private final Set<PromptTask> supportedTasks;

    public Phi3PromptStrategy(ChatTaskInstructions instructions, Set<PromptTask> supportedTasks) {
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

        StringBuilder prompt = new StringBuilder();
        if (!system.isBlank()) {
            prompt.append("<|system|>\n").append(system).append("<|end|>\n");
        }
        prompt.append("<|user|>\n").append(input.userText()).append("<|end|>\n");
        prompt.append("<|assistant|>\n");
        return prompt.toString();
    }

    private String resolveSystemTurn(PromptTask task, String systemOverride) {
        if (systemOverride != null && !systemOverride.isBlank()) {
            return systemOverride.trim();
        }
        return instructions.instructionFor(task);
    }

    private static boolean looksLikeRenderedChat(String prompt) {
        return prompt != null && prompt.contains("<|user|>") && prompt.contains("<|end|>");
    }
}
