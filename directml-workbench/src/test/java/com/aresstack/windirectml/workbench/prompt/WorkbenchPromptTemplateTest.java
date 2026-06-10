package com.aresstack.windirectml.workbench.prompt;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WorkbenchPromptTemplateTest {

    @Test
    void noneKeepsCausalUserPromptUnchanged() {
        String input = "Paste a longer text or prompt here.";

        String rendered = WorkbenchPromptTemplate.NONE.applyToUserPrompt(
                input,
                "HuggingFaceTB/SmolLM2-135M-Instruct");

        assertEquals(input, rendered);
    }

    @Test
    void translateToGermanWrapsCausalUserPrompt() {
        String input = "Paste a longer text or prompt here.";

        String rendered = WorkbenchPromptTemplate.TRANSLATE_TO_GERMAN.applyToUserPrompt(
                input,
                "HuggingFaceTB/SmolLM2-135M-Instruct");

        assertTrue(rendered.contains("Translate the text between <text> and </text> into German."));
        assertTrue(rendered.contains("<text>\n" + input + "\n</text>"));
    }

    @Test
    void translateToGermanKeepsSeq2SeqUserPromptUnchanged() {
        String input = "Paste a longer text or prompt here.";

        String rendered = WorkbenchPromptTemplate.TRANSLATE_TO_GERMAN.applyToUserPrompt(
                input,
                "google-t5/t5-small");

        assertEquals(input, rendered);
    }

    @Test
    void translateToGermanUsesT5CanonicalSystemPrompt() {
        String rendered = WorkbenchPromptTemplate.TRANSLATE_TO_GERMAN.applyToSystemPrompt(
                "",
                "google-t5/t5-small");

        assertEquals("translate English to German", rendered);
    }
}
