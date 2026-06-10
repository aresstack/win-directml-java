package com.aresstack.windirectml.workbench.prompt;

import java.util.Objects;

/**
 * User-facing prompt transformation intents for Workbench generation runs.
 * <p>
 * This enum deliberately contains no model-specific prompt text. A template is
 * only the user's intent. The model-specific {@link WorkbenchPromptProfile}
 * decides whether the intent is supported and how the final engine prompt is
 * rendered for a concrete model family.
 */
public enum WorkbenchPromptTemplate {

    NONE("Keines"),
    SUMMARIZE_DE("Zusammenfassen"),
    TRANSLATE_TO_GERMAN("Ins Deutsche übersetzen"),
    TRANSLATE_TO_ENGLISH("Ins Englische übersetzen"),
    EXPLAIN_CODE_DE("Code erklären");

    private final String displayName;

    WorkbenchPromptTemplate(String displayName) {
        this.displayName = Objects.requireNonNull(displayName, "displayName");
    }

    public boolean isNone() {
        return this == NONE;
    }

    @Override
    public String toString() {
        return displayName;
    }
}
