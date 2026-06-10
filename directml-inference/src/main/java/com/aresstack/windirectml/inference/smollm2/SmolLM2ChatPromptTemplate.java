package com.aresstack.windirectml.inference.smollm2;

import java.util.Objects;

/**
 * Renders SmolLM2-Instruct prompts with the ChatML-style template expected by the upstream model.
 */
public final class SmolLM2ChatPromptTemplate {

    private static final String IM_START = "<|im_start|>";
    private static final String IM_END = "<|im_end|>";
    private static final String DEFAULT_SYSTEM_PROMPT =
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
    private static final String GERMAN_SUMMARIZATION_ASSISTANT_PREFIX = "Zusammenfassung:\n";
    private static final String ENGLISH_SUMMARIZATION_ASSISTANT_PREFIX = "Summary:\n";

    private final String systemPrompt;

    private SmolLM2ChatPromptTemplate(String systemPrompt) {
        this.systemPrompt = Objects.requireNonNull(systemPrompt, "systemPrompt");
    }

    public static SmolLM2ChatPromptTemplate defaultInstruct() {
        return new SmolLM2ChatPromptTemplate(DEFAULT_SYSTEM_PROMPT);
    }

    public static SmolLM2ChatPromptTemplate withSystemPrompt(String systemPrompt) {
        String safePrompt = systemPrompt == null ? "" : systemPrompt.trim();
        return new SmolLM2ChatPromptTemplate(safePrompt);
    }

    public String renderUserPrompt(String userPrompt) {
        Objects.requireNonNull(userPrompt, "userPrompt");
        if (looksLikeRenderedChat(userPrompt)) {
            return userPrompt;
        }
        return renderConversation(userPrompt, "");
    }

    public String renderSummarizationPrompt(String sourceText) {
        Objects.requireNonNull(sourceText, "sourceText");
        if (looksLikeRenderedChat(sourceText)) {
            return sourceText;
        }
        String trimmedSource = sourceText.trim();
        if (looksGerman(trimmedSource)) {
            return renderConversation(renderGermanSummarizationTask(trimmedSource), GERMAN_SUMMARIZATION_ASSISTANT_PREFIX);
        }
        return renderConversation(renderEnglishSummarizationTask(trimmedSource), ENGLISH_SUMMARIZATION_ASSISTANT_PREFIX);
    }

    private static String renderGermanSummarizationTask(String sourceText) {
        return "Schreibe eine sachliche Kurzfassung des Textes zwischen <text> und </text>. "
                + "Gib nur die Kurzfassung aus. Wiederhole nicht den Quelltext. "
                + "Erfinde keine Fakten.\n\n"
                + "<text>\n"
                + sourceText
                + "\n</text>";
    }

    private static String renderEnglishSummarizationTask(String sourceText) {
        return "Write a concise factual summary of the text between <text> and </text>. "
                + "Output only the summary. Do not repeat the source text. "
                + "Do not invent facts.\n\n"
                + "<text>\n"
                + sourceText
                + "\n</text>";
    }

    private static boolean looksGerman(String text) {
        String lower = text.toLowerCase();
        return lower.contains(" der ")
                || lower.contains(" die ")
                || lower.contains(" das ")
                || lower.contains(" und ")
                || lower.contains(" nicht ")
                || lower.contains(" von ")
                || lower.contains(" im ")
                || lower.contains(" wurde ")
                || lower.contains(" werden ")
                || lower.contains(" hatte ")
                || lower.contains(" haben ")
                || lower.contains(" mit ")
                || lower.contains(" auch ")
                || lower.contains(" aber ")
                || lower.contains(" wird ")
                || lower.contains("januar")   // English: january
                || lower.contains("februar")  // English: february
                || lower.contains("juni")     // English: june
                || lower.contains("juli")     // English: july
                || lower.contains("oktober")  // English: october
                || lower.contains("dezember") // English: december
                || lower.contains("ö")
                || lower.contains("ä")
                || lower.contains("ü")
                || lower.contains("ß");
    }

    private String renderConversation(String userPrompt, String assistantPrefix) {
        StringBuilder prompt = new StringBuilder();
        if (!systemPrompt.isBlank()) {
            prompt.append(IM_START)
                    .append("system\n")
                    .append(systemPrompt)
                    .append(IM_END)
                    .append('\n');
        }
        prompt.append(IM_START)
                .append("user\n")
                .append(userPrompt)
                .append(IM_END)
                .append('\n')
                .append(IM_START)
                .append("assistant\n")
                .append(assistantPrefix);
        return prompt.toString();
    }

    public boolean looksLikeRenderedChat(String prompt) {
        return prompt != null && prompt.contains(IM_START) && prompt.contains(IM_END);
    }
}
