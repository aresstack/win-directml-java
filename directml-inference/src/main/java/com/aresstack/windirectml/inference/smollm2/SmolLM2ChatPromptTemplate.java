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

    public static SmolLM2ChatPromptTemplate withSystemPrompt(String systemPrompt) {
        String safePrompt = systemPrompt == null ? "" : systemPrompt.trim();
        if (safePrompt.isEmpty()) {
            return defaultInstruct();
        }
        return new SmolLM2ChatPromptTemplate(safePrompt);
    }

    public String renderUserPrompt(String userPrompt) {
        Objects.requireNonNull(userPrompt, "userPrompt");
        if (looksLikeRenderedChat(userPrompt)) {
            return userPrompt;
        }
        return renderConversation(userPrompt);
    }

    private String renderConversation(String userPrompt) {
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
                .append("assistant\n");
        return prompt.toString();
    }

    public boolean looksLikeRenderedChat(String prompt) {
        return prompt != null && prompt.contains(IM_START) && prompt.contains(IM_END);
    }
}
