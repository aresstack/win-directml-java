package com.aresstack.windirectml.inference.prompt;

import java.util.Objects;
import java.util.Set;

/**
 * Prompt strategy for Gemma 3 instruction/chat checkpoints.
 *
 * <p>Gemma 3 instruction models use turn markers instead of ChatML. The workbench
 * keeps task instructions close to the assistant turn because tiny instruction
 * models are strongly recency-biased.</p>
 */
public final class Gemma3PromptStrategy implements PromptStrategy {

    private static final String START_OF_TURN = "<start_of_turn>";
    private static final String END_OF_TURN = "<end_of_turn>";

    private final ChatTaskInstructions instructions;
    private final Set<PromptTask> supportedTasks;

    public Gemma3PromptStrategy(ChatTaskInstructions instructions, Set<PromptTask> supportedTasks) {
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
        if (looksLikeRenderedGemmaPrompt(input.userText())) {
            return input.userText();
        }
        PromptTask task = supportedTasks.contains(input.task()) ? input.task() : PromptTask.NONE;
        String userTurn = composeUserTurn(task, input.systemOverride(), input.userText());
        return START_OF_TURN + "user\n"
                + userTurn
                + END_OF_TURN + "\n"
                + START_OF_TURN + "model\n";
    }

    private String composeUserTurn(PromptTask task, String systemOverride, String userText) {
        String text = userText == null ? "" : userText;
        StringBuilder result = new StringBuilder();
        if (systemOverride != null && !systemOverride.isBlank()) {
            result.append(systemOverride.trim()).append("\n\n");
        }
        result.append(text);
        if (systemOverride == null || systemOverride.isBlank()) {
            String instruction = instructions.instructionFor(task);
            if (!instruction.isBlank()) {
                result.append("\n\n").append(instruction);
            }
        }
        return result.toString();
    }

    private static boolean looksLikeRenderedGemmaPrompt(String prompt) {
        return prompt != null && prompt.contains(START_OF_TURN) && prompt.contains(END_OF_TURN);
    }
}
