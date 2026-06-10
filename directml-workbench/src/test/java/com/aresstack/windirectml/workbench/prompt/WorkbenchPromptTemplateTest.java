package com.aresstack.windirectml.workbench.prompt;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WorkbenchPromptTemplateTest {

    @Test
    void noneKeepsSmolUserPromptUnchanged() {
        String input = "Paste a longer text or prompt here.";

        RenderedWorkbenchPrompt rendered = WorkbenchPromptProfileRegistry.render(
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                WorkbenchPromptTemplate.NONE,
                input);

        assertEquals(input, rendered.getUserPrompt());
        assertEquals("", rendered.getSystemPrompt());
    }

    @Test
    void smolTranslationUsesShortTaskWithoutXmlBoundary() {
        String input = "Paste a longer text or prompt here.";

        RenderedWorkbenchPrompt rendered = WorkbenchPromptProfileRegistry.render(
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                WorkbenchPromptTemplate.TRANSLATE_TO_GERMAN,
                input);

        assertTrue(rendered.getUserPrompt().startsWith("Translate to German."));
        assertTrue(rendered.getUserPrompt().contains(input));
        assertFalse(rendered.getUserPrompt().contains("<text>"));
        assertFalse(rendered.getUserPrompt().contains("</text>"));
        assertEquals("", rendered.getSystemPrompt());
    }

    @Test
    void googleT5TranslationUsesCanonicalPrefix() {
        String input = "Paste a longer text or prompt here.";

        RenderedWorkbenchPrompt rendered = WorkbenchPromptProfileRegistry.render(
                "google-t5/t5-small",
                WorkbenchPromptTemplate.TRANSLATE_TO_GERMAN,
                input);

        assertEquals("translate English to German: " + input, rendered.getUserPrompt());
        assertEquals("", rendered.getSystemPrompt());
    }

    @Test
    void flanT5TranslationUsesNaturalInstruction() {
        String input = "Paste a longer text or prompt here.";

        RenderedWorkbenchPrompt rendered = WorkbenchPromptProfileRegistry.render(
                "google/flan-t5-small",
                WorkbenchPromptTemplate.TRANSLATE_TO_GERMAN,
                input);

        assertTrue(rendered.getUserPrompt().startsWith("Translate the following text to German:"));
        assertTrue(rendered.getUserPrompt().contains(input));
        assertEquals("", rendered.getSystemPrompt());
    }

    @Test
    void codeT5SmallDoesNotOfferTranslation() {
        List<WorkbenchPromptTemplate> templates = WorkbenchPromptProfileRegistry.templatesFor("Salesforce/codet5-small");

        assertTrue(templates.contains(WorkbenchPromptTemplate.NONE));
        assertTrue(templates.contains(WorkbenchPromptTemplate.EXPLAIN_CODE_DE));
        assertFalse(templates.contains(WorkbenchPromptTemplate.TRANSLATE_TO_GERMAN));
    }

    @Test
    void codeT5MultiSumOnlyOffersSummaryIntent() {
        List<WorkbenchPromptTemplate> templates = WorkbenchPromptProfileRegistry.templatesFor("Salesforce/codet5-base-multi-sum");

        assertTrue(templates.contains(WorkbenchPromptTemplate.NONE));
        assertTrue(templates.contains(WorkbenchPromptTemplate.SUMMARIZE_DE));
        assertFalse(templates.contains(WorkbenchPromptTemplate.TRANSLATE_TO_GERMAN));
        assertFalse(templates.contains(WorkbenchPromptTemplate.EXPLAIN_CODE_DE));
    }

    @Test
    void qwenCodeExplainTransformsUserPromptOnly() {
        String input = "class Example {}";

        RenderedWorkbenchPrompt rendered = WorkbenchPromptProfileRegistry.render(
                "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                WorkbenchPromptTemplate.EXPLAIN_CODE_DE,
                input);

        assertTrue(rendered.getUserPrompt().startsWith("Erkläre den folgenden Code"));
        assertTrue(rendered.getUserPrompt().contains(input));
        assertEquals("", rendered.getSystemPrompt());
    }
}
