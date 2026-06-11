package com.aresstack.windirectml.inference.prompt;

import java.util.Set;

/**
 * Model-family specific prompt renderer (the Strategy in the prompt pipeline).
 *
 * <p>A strategy is the single source of truth for one model family. It decides:
 * <ul>
 *   <li>which {@link PromptTask}s the family can meaningfully serve
 *       ({@link #supportedTasks()}), and</li>
 *   <li>how a model-neutral {@link PromptInput} becomes the exact wire string the
 *       model was trained on ({@link #renderPrompt(PromptInput)}).</li>
 * </ul>
 *
 * <p>Strategies live in the inference layer, never in the Workbench. The
 * Workbench is a test client that only selects a {@link PromptTask} and supplies
 * raw text.</p>
 */
public interface PromptStrategy {

    /**
     * Render the final, tokenizer-ready prompt string for this model family.
     *
     * <p>Unsupported tasks must degrade gracefully to {@link PromptTask#NONE}
     * behaviour rather than throwing.</p>
     */
    String renderPrompt(PromptInput input);

    /**
     * Tasks this strategy renders in a model-appropriate way. Callers (e.g. the
     * Workbench UI) use this to offer only meaningful options. Always contains
     * {@link PromptTask#NONE}.
     */
    Set<PromptTask> supportedTasks();

    default String renderPrompt(PromptTask task, String userText) {
        return renderPrompt(PromptInput.of(task, userText));
    }
}
