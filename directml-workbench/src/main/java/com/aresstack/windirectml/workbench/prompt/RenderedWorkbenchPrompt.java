package com.aresstack.windirectml.workbench.prompt;

import java.util.Objects;

/**
 * Final prompt payload handed to an inference engine.
 * <p>
 * The Workbench prompt pipeline resolves all selected templates before runtime
 * execution. Engines receive only this already-rendered prompt payload and do
 * not interpret Workbench template selections.
 */
public final class RenderedWorkbenchPrompt {

    private final String userPrompt;
    private final String systemPrompt;

    public RenderedWorkbenchPrompt(String userPrompt, String systemPrompt) {
        this.userPrompt = Objects.requireNonNull(userPrompt, "userPrompt");
        this.systemPrompt = Objects.requireNonNull(systemPrompt, "systemPrompt");
    }

    public static RenderedWorkbenchPrompt userPromptOnly(String userPrompt) {
        return new RenderedWorkbenchPrompt(userPrompt == null ? "" : userPrompt, "");
    }

    public String getUserPrompt() {
        return userPrompt;
    }

    public String getSystemPrompt() {
        return systemPrompt;
    }
}
