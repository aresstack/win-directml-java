package com.aresstack.windirectml.workbench.prompt;

import java.util.List;

/**
 * Model-family specific prompt renderer.
 * <p>
 * A profile turns a Workbench template intent and the user's raw input into the
 * exact prompt text expected by one model family. Unsupported templates are not
 * shown in the UI for that model.
 */
public interface WorkbenchPromptProfile {

    List<WorkbenchPromptTemplate> supportedTemplates();

    RenderedWorkbenchPrompt render(WorkbenchPromptTemplate template, String inputText);
}
