package com.aresstack.windirectml.inference.prompt;

import java.util.Set;

/**
 * Pass-through {@link PromptStrategy} for models whose wire format is unknown or
 * that should receive the user text verbatim. Supports only {@link PromptTask#NONE}.
 */
public final class RawPromptStrategy implements PromptStrategy {

    public static final RawPromptStrategy INSTANCE = new RawPromptStrategy();

    private RawPromptStrategy() {
    }

    @Override
    public String renderPrompt(PromptInput input) {
        return input == null ? "" : input.userText();
    }

    @Override
    public Set<PromptTask> supportedTasks() {
        return Set.of(PromptTask.NONE);
    }
}
