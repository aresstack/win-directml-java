package com.aresstack.windirectml.inference.prompt;

import java.util.Objects;

/**
 * Model-neutral input handed to a {@link PromptStrategy}.
 *
 * <p>Combines the user's raw text, the requested {@link PromptTask} and an
 * optional free-form system instruction. The strategy decides how (and whether)
 * each part is rendered for its model family.</p>
 *
 * <ul>
 *   <li>{@code task} — the intent (translate, summarize, ...). Never a prompt string.</li>
 *   <li>{@code userText} — the raw text typed by the user; never pre-wrapped.</li>
 *   <li>{@code systemOverride} — optional caller-supplied system instruction that
 *       takes precedence over the task's default instruction. Blank means "none".</li>
 * </ul>
 */
public record PromptInput(PromptTask task, String userText, String systemOverride) {

    public PromptInput {
        task = task == null ? PromptTask.NONE : task;
        userText = userText == null ? "" : userText;
        systemOverride = systemOverride == null ? "" : systemOverride;
    }

    public static PromptInput of(PromptTask task, String userText) {
        return new PromptInput(task, userText, "");
    }

    public static PromptInput raw(String userText) {
        return new PromptInput(PromptTask.NONE, userText, "");
    }

    public boolean hasSystemOverride() {
        return systemOverride != null && !systemOverride.isBlank();
    }
}
