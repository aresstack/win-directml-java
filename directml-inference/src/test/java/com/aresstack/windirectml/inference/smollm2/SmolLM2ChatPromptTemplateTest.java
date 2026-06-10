package com.aresstack.windirectml.inference.smollm2;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
    void rendersGermanSummarizationPromptWithTextBoundaryAndAssistantPrefix() {
        SmolLM2ChatPromptTemplate template = SmolLM2ChatPromptTemplate.defaultInstruct();

        String rendered = template.renderSummarizationPrompt("Adolf Hitler wurde am 30. Januar 1933 ernannt.");

        assertTrue(rendered.startsWith("<|im_start|>system\n"));
        assertTrue(rendered.contains("Schreibe eine sachliche Kurzfassung"));
        assertTrue(rendered.contains("Gib nur die Kurzfassung aus."));
        assertTrue(rendered.contains("<text>\nAdolf Hitler wurde am 30. Januar 1933 ernannt.\n</text>"));
        assertTrue(rendered.endsWith("<|im_start|>assistant\nZusammenfassung:\n"));
    }

    @Test
    void rendersEnglishSummarizationPromptWithEnglishInstruction() {
        SmolLM2ChatPromptTemplate template = SmolLM2ChatPromptTemplate.defaultInstruct();

        String rendered = template.renderSummarizationPrompt("Paste a longer text or prompt here.");

        assertTrue(rendered.contains("Write a concise factual summary"));
        assertTrue(rendered.contains("Output only the summary."));
        assertTrue(rendered.contains("<text>\nPaste a longer text or prompt here.\n</text>"));
        assertTrue(rendered.endsWith("<|im_start|>assistant\nSummary:\n"));
    }

    @Test
    void keepsAlreadyRenderedSummarizationPromptUnchanged() {
        SmolLM2ChatPromptTemplate template = SmolLM2ChatPromptTemplate.defaultInstruct();
        String prompt = "<|im_start|>user\nBitte zusammenfassen<|im_end|>\n<|im_start|>assistant\n";

        assertEquals(prompt, template.renderSummarizationPrompt(prompt));
    }
}
