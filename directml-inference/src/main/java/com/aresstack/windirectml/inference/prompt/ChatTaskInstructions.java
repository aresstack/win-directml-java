package com.aresstack.windirectml.inference.prompt;

import java.util.Map;
import java.util.Objects;

/**
 * Maps a neutral {@link PromptTask} to a natural-language system instruction for
 * instruction-tuned chat models (SmolLM2, Qwen, Phi-3, ...).
 *
 * <p>These instructions are deliberately short, English and imperative: small
 * instruction models follow a concise directive in the system turn far more
 * reliably than a verbose or localized one. {@link PromptTask#NONE} maps to no
 * instruction at all.</p>
 *
 * <p>This is shared by every chat-style {@link PromptStrategy}; non-chat models
 * such as T5 do not use it (they render canonical task prefixes instead).</p>
 */
public final class ChatTaskInstructions {

    private static final Map<PromptTask, String> STANDARD = Map.of(
            PromptTask.SUMMARIZE,
            "Summarize the user's text. Reply in the same language as the text. Output only the summary.",
            PromptTask.TRANSLATE_TO_GERMAN,
            "Translate the user's text into German. Output only the translation.",
            PromptTask.TRANSLATE_TO_ENGLISH,
            "Translate the user's text into English. Output only the translation.",
            PromptTask.EXPLAIN_CODE,
            "Explain the user's code concisely in German.");

    private final Map<PromptTask, String> instructions;

    private ChatTaskInstructions(Map<PromptTask, String> instructions) {
        this.instructions = Map.copyOf(instructions);
    }

    public static ChatTaskInstructions standard() {
        return new ChatTaskInstructions(STANDARD);
    }

    public static ChatTaskInstructions of(Map<PromptTask, String> instructions) {
        return new ChatTaskInstructions(Objects.requireNonNull(instructions, "instructions"));
    }

    /**
     * Returns the system instruction for {@code task}, or an empty string for
     * {@link PromptTask#NONE} or any task without a mapping.
     */
    public String instructionFor(PromptTask task) {
        if (task == null || task == PromptTask.NONE) {
            return "";
        }
        return instructions.getOrDefault(task, "");
    }
}
