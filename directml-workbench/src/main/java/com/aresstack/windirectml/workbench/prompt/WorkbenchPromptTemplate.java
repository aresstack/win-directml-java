package com.aresstack.windirectml.workbench.prompt;

import com.aresstack.windirectml.config.generation.GenerationModelRegistry;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * User-selectable prompt templates for Workbench generation runs.
 * <p>
 * Templates are an application-layer convenience. They never alter the source
 * text directly. Instead, they add an optional task instruction that is passed
 * through the normal system-prompt channel. This keeps runtimes model-neutral:
 * causal chat models receive the instruction as system text, while T5-family
 * engines can map the same instruction to their text-to-text task prefix.
 */
public enum WorkbenchPromptTemplate {

    NONE("Keines", "", true, true),
    SUMMARIZE_DE("Zusammenfassen", "Fasse den folgenden Text kurz und sachlich auf Deutsch zusammen. Gib nur die Zusammenfassung aus.", true, true),
    TRANSLATE_TO_GERMAN("Ins Deutsche übersetzen", "Übersetze den folgenden Text ins Deutsche. Gib nur die Übersetzung aus.", true, true),
    TRANSLATE_TO_ENGLISH("Ins Englische übersetzen", "Translate the following text into English. Output only the translation.", true, true),
    EXPLAIN_CODE_DE("Code erklären", "Erkläre den folgenden Code kurz und verständlich auf Deutsch.", true, false);

    private final String displayName;
    private final String instruction;
    private final boolean causalLmSupported;
    private final boolean seq2SeqSupported;

    WorkbenchPromptTemplate(String displayName,
                            String instruction,
                            boolean causalLmSupported,
                            boolean seq2SeqSupported) {
        this.displayName = Objects.requireNonNull(displayName, "displayName");
        this.instruction = Objects.requireNonNull(instruction, "instruction");
        this.causalLmSupported = causalLmSupported;
        this.seq2SeqSupported = seq2SeqSupported;
    }

    public static List<WorkbenchPromptTemplate> templatesFor(String modelId) {
        GenerationModelRegistry.Entry entry = GenerationModelRegistry.findByModelId(modelId);
        GenerationModelRegistry.Architecture architecture = entry == null
                ? GenerationModelRegistry.Architecture.CAUSAL_LM
                : entry.architecture();
        List<WorkbenchPromptTemplate> templates = new ArrayList<WorkbenchPromptTemplate>();
        for (WorkbenchPromptTemplate template : values()) {
            if (template.supports(architecture, modelId)) {
                templates.add(template);
            }
        }
        return Collections.unmodifiableList(templates);
    }

    public String applyToSystemPrompt(String userSystemPrompt) {
        String customPrompt = userSystemPrompt == null ? "" : userSystemPrompt.trim();
        if (instruction.isEmpty()) {
            return customPrompt;
        }
        if (customPrompt.isEmpty()) {
            return instruction;
        }
        return customPrompt + "\n\n" + instruction;
    }

    public boolean isNone() {
        return this == NONE;
    }

    @Override
    public String toString() {
        return displayName;
    }

    private boolean supports(GenerationModelRegistry.Architecture architecture, String modelId) {
        if (this == NONE) {
            return true;
        }
        if (architecture == GenerationModelRegistry.Architecture.CAUSAL_LM) {
            return causalLmSupported;
        }
        if (architecture == GenerationModelRegistry.Architecture.SEQ2SEQ) {
            return seq2SeqSupported || isCodeT5Model(modelId);
        }
        return false;
    }

    private static boolean isCodeT5Model(String modelId) {
        return modelId != null && modelId.toLowerCase(Locale.ROOT).contains("codet5");
    }
}
