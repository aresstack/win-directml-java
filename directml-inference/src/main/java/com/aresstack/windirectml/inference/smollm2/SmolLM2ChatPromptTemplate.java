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

    private final String systemPrompt;

    private SmolLM2ChatPromptTemplate(String systemPrompt) {
        this.systemPrompt = Objects.requireNonNull(systemPrompt, "systemPrompt");
    }

    public static SmolLM2ChatPromptTemplate defaultInstruct() {
        return new SmolLM2ChatPromptTemplate(DEFAULT_SYSTEM_PROMPT);
    }

    public String renderUserPrompt(String userPrompt) {
        Objects.requireNonNull(userPrompt, "userPrompt");
        if (looksLikeRenderedChat(userPrompt)) {
            return userPrompt;
        }
        return renderConversation(userPrompt);
    }

    public String renderSummarizationPrompt(String sourceText) {
        Objects.requireNonNull(sourceText, "sourceText");
        if (looksLikeRenderedChat(sourceText)) {
            return sourceText;
        }
        String taskPrompt = "Aufgabe: Fasse den folgenden Quelltext sachlich und knapp zusammen.\n"
                + "Regeln:\n"
                + "- Antworte in derselben Sprache wie der Quelltext.\n"
                + "- Verwende ausschließlich Informationen aus dem Quelltext.\n"
                + "- Erfinde keine zusätzlichen Fakten, Übersetzungen, Rollen, Datumsangaben oder Ämter.\n"
                + "- Übernimm Namen, Orte, Datumsangaben und historische Fakten nur aus dem Quelltext.\n"
                + "- Wenn der Quelltext deutsch ist, antworte deutsch.\n\n"
                + "Quelltext:\n<<<\n"
                + sourceText
                + "\n>>>\n\n"
                + "Zusammenfassung:";
        return renderConversation(taskPrompt);
    }

    private String renderConversation(String userPrompt) {
        return IM_START + "system\n"
                + systemPrompt
                + IM_END + "\n"
                + IM_START + "user\n"
                + userPrompt
                + IM_END + "\n"
                + IM_START + "assistant\n";
    }

    public boolean looksLikeRenderedChat(String prompt) {
        return prompt != null && prompt.contains(IM_START) && prompt.contains(IM_END);
    }
}
