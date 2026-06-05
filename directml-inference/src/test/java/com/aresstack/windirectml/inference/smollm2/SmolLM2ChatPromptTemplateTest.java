package com.aresstack.windirectml.inference.smollm2;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

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
    void keepsAlreadyRenderedChatPromptUnchanged() {
        SmolLM2ChatPromptTemplate template = SmolLM2ChatPromptTemplate.defaultInstruct();
        String prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";

        assertEquals(prompt, template.renderUserPrompt(prompt));
    }
}
