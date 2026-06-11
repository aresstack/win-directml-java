package com.aresstack.windirectml.inference.prompt;

import java.util.EnumSet;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * {@link PromptStrategy} for T5-family text-to-text models (google-t5, flan-t5,
 * CodeT5, ...).
 *
 * <p>Base T5 checkpoints are <em>not</em> chat models. They were trained on
 * canonical task prefixes such as {@code summarize:} or
 * {@code translate English to German:}. This strategy maps a neutral
 * {@link PromptTask} to the prefix configured for the concrete model and emits
 * {@code "<prefix>: <text>"}. {@link PromptTask#NONE} (or a task without a
 * configured prefix) passes the text through unchanged.</p>
 *
 * <p>Because the prefix is essential for usable T5 output, this template lives in
 * the inference layer next to the engine, never in the Workbench client.</p>
 */
public final class T5PromptStrategy implements PromptStrategy {

    private final Map<PromptTask, String> taskPrefixes;
    private final Set<PromptTask> supportedTasks;

    public T5PromptStrategy(Map<PromptTask, String> taskPrefixes) {
        this.taskPrefixes = Map.copyOf(Objects.requireNonNull(taskPrefixes, "taskPrefixes"));
        EnumSet<PromptTask> tasks = EnumSet.of(PromptTask.NONE);
        tasks.addAll(this.taskPrefixes.keySet());
        this.supportedTasks = Set.copyOf(tasks);
    }

    @Override
    public Set<PromptTask> supportedTasks() {
        return supportedTasks;
    }

    @Override
    public String renderPrompt(PromptInput input) {
        Objects.requireNonNull(input, "input");
        String text = input.userText() == null ? "" : input.userText().trim();
        if (text.isEmpty()) {
            return text;
        }
        if (hasKnownTaskPrefix(text)) {
            return text;
        }
        String prefix = taskPrefixes.get(input.task());
        if (prefix == null || prefix.isBlank()) {
            return text;
        }
        return prefix + ": " + text;
    }

    private static boolean hasKnownTaskPrefix(String text) {
        String lower = text.toLowerCase(Locale.ROOT);
        return lower.startsWith("summarize:")
                || lower.startsWith("translate ")
                || lower.startsWith("explain:")
                || lower.startsWith("explain java:")
                || lower.startsWith("summarize java:")
                || lower.startsWith("classify:");
    }
}
