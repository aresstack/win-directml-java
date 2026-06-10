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
 * Templates are an application-layer convenience. They keep the source text
 * unchanged and only add task metadata around it. Decoder-only chat models
 * receive natural-language system instructions. T5-style seq2seq models receive
 * canonical text-to-text task prefixes, because natural chat instructions make
 * base T5 models behave like span-infilling models.
 */
public enum WorkbenchPromptTemplate {

    NONE("Keines", "", "", true, true),
    SUMMARIZE_DE("Zusammenfassen",
            "Fasse den folgenden Text kurz und sachlich auf Deutsch zusammen. Gib nur die Zusammenfassung aus.",
            "summarize",
            true,
            true),
    TRANSLATE_TO_GERMAN("Ins Deutsche übersetzen",
            "Übersetze den folgenden Text ins Deutsche. Gib nur die Übersetzung aus.",
            "translate English to German",
            true,
            true),
    TRANSLATE_TO_ENGLISH("Ins Englische übersetzen",
            "Translate the following text into English. Output only the translation.",
            "translate German to English",
            true,
            true),
    EXPLAIN_CODE_DE("Code erklären",
            "Erkläre den folgenden Code kurz und verständlich auf Deutsch.",
            "explain java",
            true,
            false);

    private final String displayName;
    private final String causalInstruction;
    private final String seq2SeqInstruction;
    private final boolean causalLmSupported;
    private final boolean seq2SeqSupported;

    WorkbenchPromptTemplate(String displayName,
                            String causalInstruction,
                            String seq2SeqInstruction,
                            boolean causalLmSupported,
                            boolean seq2SeqSupported) {
        this.displayName = Objects.requireNonNull(displayName, "displayName");
        this.causalInstruction = Objects.requireNonNull(causalInstruction, "causalInstruction");
        this.seq2SeqInstruction = Objects.requireNonNull(seq2SeqInstruction, "seq2SeqInstruction");
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

    public String applyToSystemPrompt(String userSystemPrompt, String modelId) {
        GenerationModelRegistry.Entry entry = GenerationModelRegistry.findByModelId(modelId);
        GenerationModelRegistry.Architecture architecture = entry == null
                ? GenerationModelRegistry.Architecture.CAUSAL_LM
                : entry.architecture();
        if (architecture == GenerationModelRegistry.Architecture.SEQ2SEQ) {
            return applyToSeq2SeqPrompt(userSystemPrompt);
        }
        return applyToCausalPrompt(userSystemPrompt);
    }

    public String applyToSystemPrompt(String userSystemPrompt) {
        return applyToCausalPrompt(userSystemPrompt);
    }

    public boolean isNone() {
        return this == NONE;
    }

    @Override
    public String toString() {
        return displayName;
    }

    private String applyToCausalPrompt(String userSystemPrompt) {
        String customPrompt = sanitize(userSystemPrompt);
        if (causalInstruction.isEmpty()) {
            return customPrompt;
        }
        if (customPrompt.isEmpty()) {
            return causalInstruction;
        }
        return customPrompt + "\n\n" + causalInstruction;
    }

    private String applyToSeq2SeqPrompt(String userSystemPrompt) {
        String customPrompt = sanitize(userSystemPrompt);
        if (seq2SeqInstruction.isEmpty()) {
            return customPrompt;
        }
        if (customPrompt.isEmpty()) {
            return seq2SeqInstruction;
        }
        return seq2SeqInstruction + "\n" + customPrompt;
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

    private static String sanitize(String value) {
        return value == null ? "" : value.trim();
    }

    private static boolean isCodeT5Model(String modelId) {
        return modelId != null && modelId.toLowerCase(Locale.ROOT).contains("codet5");
    }
}
