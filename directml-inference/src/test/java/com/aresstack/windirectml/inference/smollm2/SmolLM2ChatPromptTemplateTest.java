package com.aresstack.windirectml.inference.smollm2;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SmolLM2ChatPromptTemplateTest {

    @Test
    void rendersPlainPromptAsSmolLm2InstructConversation() {
        SmolLM2ChatPromptTemplate template = SmolLM2ChatPromptTemplate.defaultInstruct();

        String rendered = template.renderUserPrompt("Summarize this text.");

        assertTrue(rendered.startsWith("<|im_start|>system\n"));
        assertTrue(rendered.contains("<|im_start|>user\nSummarize this text.<|im_end|>\n"));
        assertTrue(rendered.endsWith("<|im_start|>assistant\n"));
    }

    @Test
    void omitsSystemBlockWhenSystemPromptIsBlank() {
        SmolLM2ChatPromptTemplate template = SmolLM2ChatPromptTemplate.withSystemPrompt("");

        String rendered = template.renderUserPrompt("Paste a longer text or prompt here.");

        assertTrue(rendered.startsWith("<|im_start|>user\n"));
        assertTrue(rendered.contains("Paste a longer text or prompt here."));
        assertTrue(rendered.endsWith("<|im_start|>assistant\n"));
    }

    @Test
    void attachesExplicitSystemPromptWithoutChangingUserText() {
        SmolLM2ChatPromptTemplate template = SmolLM2ChatPromptTemplate.withSystemPrompt("fasse den Text zusammen");

        String rendered = template.renderUserPrompt("Adolf Hitler wurde am 30. Januar 1933 ernannt.");

        assertTrue(rendered.startsWith("<|im_start|>system\n"));
        assertTrue(rendered.contains("fasse den Text zusammen"));
        assertTrue(rendered.contains("<|im_start|>user\nAdolf Hitler wurde am 30. Januar 1933 ernannt.<|im_end|>\n"));
    }

    @Test
    void keepsAlreadyRenderedChatPromptUnchanged() {
        SmolLM2ChatPromptTemplate template = SmolLM2ChatPromptTemplate.defaultInstruct();
        String prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";

        assertEquals(prompt, template.renderUserPrompt(prompt));
    }

    @Test
    void doesNotContainLegacyTextBoundaryPrompting() {
        SmolLM2ChatPromptTemplate template = SmolLM2ChatPromptTemplate.withSystemPrompt("");

        String rendered = template.renderUserPrompt("Translate to German.\n\nPaste a longer text or prompt here.");

        assertFalse(rendered.contains("<text>"));
        assertFalse(rendered.contains("</text>"));
        assertFalse(rendered.contains("Zusammenfassung:"));
        assertFalse(rendered.contains("Summary:"));
    }
}
