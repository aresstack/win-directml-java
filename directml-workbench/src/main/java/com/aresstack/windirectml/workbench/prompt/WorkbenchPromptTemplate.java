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
 * Templates are an application-layer convenience. The {@link #NONE} template
 * passes the source text unchanged. Task templates intentionally wrap the user
 * text for causal language models, because small instruction models follow a
 * concrete user task more reliably than a system-only instruction. T5-style
 * seq2seq models receive canonical task prefixes through the system prompt so
 * the family-specific T5 formatter can render {@code summarize: ...} or
 * {@code translate English to German: ...}.
 */
public enum WorkbenchPromptTemplate {

    NONE("Keines", "", "", CausalTask.NONE, true, true),
    SUMMARIZE_DE("Zusammenfassen",
            "Fasse den folgenden Text kurz und sachlich auf Deutsch zusammen. Gib nur die Zusammenfassung aus.",
            "summarize",
            CausalTask.SUMMARIZE_GERMAN,
            true,
            true),
    TRANSLATE_TO_GERMAN("Ins Deutsche übersetzen",
            "Translate the user's text into German. Output only the translation.",
            "translate English to German",
            CausalTask.TRANSLATE_TO_GERMAN,
            true,
            true),
    TRANSLATE_TO_ENGLISH("Ins Englische übersetzen",
            "Translate the user's text into English. Output only the translation.",
            "translate German to English",
            CausalTask.TRANSLATE_TO_ENGLISH,
            true,
            true),
    EXPLAIN_CODE_DE("Code erklären",
            "Explain the user's code in concise German.",
            "explain java",
            CausalTask.EXPLAIN_CODE_GERMAN,
            true,
            false);

    private final String displayName;
    private final String causalInstruction;
    private final String seq2SeqInstruction;
    private final CausalTask causalTask;
    private final boolean causalLmSupported;
    private final boolean seq2SeqSupported;

    WorkbenchPromptTemplate(String displayName,
                            String causalInstruction,
                            String seq2SeqInstruction,
                            CausalTask causalTask,
                            boolean causalLmSupported,
                            boolean seq2SeqSupported) {
        this.displayName = Objects.requireNonNull(displayName, "displayName");
        this.causalInstruction = Objects.requireNonNull(causalInstruction, "causalInstruction");
        this.seq2SeqInstruction = Objects.requireNonNull(seq2SeqInstruction, "seq2SeqInstruction");
        this.causalTask = Objects.requireNonNull(causalTask, "causalTask");
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

    public String applyToUserPrompt(String userPrompt, String modelId) {
        GenerationModelRegistry.Entry entry = GenerationModelRegistry.findByModelId(modelId);
        GenerationModelRegistry.Architecture architecture = entry == null
                ? GenerationModelRegistry.Architecture.CAUSAL_LM
                : entry.architecture();
        String safePrompt = userPrompt == null ? "" : userPrompt;
        if (architecture == GenerationModelRegistry.Architecture.SEQ2SEQ || isNone()) {
            return safePrompt;
        }
        return causalTask.render(safePrompt);
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

    private enum CausalTask {
        NONE {
            @Override
            String render(String input) {
                return input;
            }
        },
        SUMMARIZE_GERMAN {
            @Override
            String render(String input) {
                return renderTextTask(
                        "Summarize the text between <text> and </text> in concise German. Output only the summary.",
                        "text",
                        input);
            }
        },
        TRANSLATE_TO_GERMAN {
            @Override
            String render(String input) {
                return renderTextTask(
                        "Translate the text between <text> and </text> into German. Output only the translation.",
                        "text",
                        input);
            }
        },
        TRANSLATE_TO_ENGLISH {
            @Override
            String render(String input) {
                return renderTextTask(
                        "Translate the text between <text> and </text> into English. Output only the translation.",
                        "text",
                        input);
            }
        },
        EXPLAIN_CODE_GERMAN {
            @Override
            String render(String input) {
                return renderTextTask(
                        "Explain the code between <code> and </code> in concise German.",
                        "code",
                        input);
            }
        };

        abstract String render(String input);

        private static String renderTextTask(String instruction, String tagName, String input) {
            String safeInput = input == null ? "" : input;
            return instruction
                    + "\n\n<" + tagName + ">\n"
                    + safeInput
                    + "\n</" + tagName + ">";
        }
    }
}
