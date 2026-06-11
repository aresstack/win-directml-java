package com.aresstack.windirectml.workbench.prompt;

import com.aresstack.windirectml.inference.prompt.PromptTask;

import java.util.Map;

/**
 * German UI labels for the model-neutral {@link PromptTask} values.
 *
 * <p>Labels are a pure Workbench (test-client) concern. The inference layer only
 * knows the neutral {@link PromptTask} enum; the human-readable presentation
 * lives here so the engine stays free of UI text.</p>
 */
public final class PromptTaskLabels {

    private static final Map<PromptTask, String> LABELS = Map.of(
            PromptTask.NONE, "Keines",
            PromptTask.SUMMARIZE, "Zusammenfassen",
            PromptTask.TRANSLATE_TO_GERMAN, "Ins Deutsche übersetzen",
            PromptTask.TRANSLATE_TO_ENGLISH, "Ins Englische übersetzen",
            PromptTask.EXPLAIN_CODE, "Code erklären");

    private PromptTaskLabels() {
    }

    public static String labelFor(PromptTask task) {
        if (task == null) {
            return LABELS.get(PromptTask.NONE);
        }
        return LABELS.getOrDefault(task, task.name());
    }
}
