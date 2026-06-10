package com.aresstack.windirectml.inference.smollm2;

import java.util.Objects;

/**
 * Renders the text that is passed to the SmolLM2 tokenizer.
 */
@FunctionalInterface
public interface SmolLM2PromptRenderer {

    SmolLM2PromptRenderer RAW_INPUT = new SmolLM2PromptRenderer() {
        @Override
        public String render(String prompt) {
            return prompt == null ? "" : prompt;
        }
    };

    String render(String prompt);

    static SmolLM2PromptRenderer rawInput() {
        return RAW_INPUT;
    }

    static SmolLM2PromptRenderer chatTemplate(SmolLM2ChatPromptTemplate template) {
        Objects.requireNonNull(template, "template");
        return template::renderUserPrompt;
    }
}
