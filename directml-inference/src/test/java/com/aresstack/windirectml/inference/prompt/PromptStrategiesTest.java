package com.aresstack.windirectml.inference.prompt;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Behavioural contract for the model-neutral prompt pipeline and the per-family
 * {@link PromptStrategy} rendering.
 */
class PromptStrategiesTest {

    // ---- SmolLM2 / ChatML -------------------------------------------------

    @Test
    void smolLm2NoneKeepsUserTextRawInChatMlUserTurn() {
        String input = "Paste a longer text or prompt here.";

        String prompt = PromptStrategies.forModel("HuggingFaceTB/SmolLM2-135M-Instruct")
                .renderPrompt(PromptInput.of(PromptTask.NONE, input));

        // No system turn for NONE, no hard-coded default system prompt.
        assertFalse(prompt.contains("<|im_start|>system"));
        assertTrue(prompt.contains("<|im_start|>user\n" + input + "<|im_end|>\n"));
        assertTrue(prompt.endsWith("<|im_start|>assistant\n"));
        assertFalse(prompt.contains("You are a helpful AI assistant"));
    }

    @Test
    void smolLm2TranslationPutsTaskInUserTurnAfterText() {
        String input = "The quick brown fox.";

        String prompt = PromptStrategies.forModel("HuggingFaceTB/SmolLM2-360M-Instruct")
                .renderPrompt(PromptInput.of(PromptTask.TRANSLATE_TO_GERMAN, input));

        // No system turn: the task instruction is appended AFTER the user text
        // (recency) so tiny instruction models act on it instead of echoing.
        assertFalse(prompt.contains("<|im_start|>system"));
        assertTrue(prompt.contains("<|im_start|>user\n" + input + "\n\nTranslate the user's text into German."));
        assertTrue(prompt.endsWith("<|im_start|>assistant\n"));
    }

    @Test
    void chatMlSystemOverrideTakesPrecedenceOverTaskInstruction() {
        String prompt = PromptStrategies.forModel("HuggingFaceTB/SmolLM2-135M-Instruct")
                .renderPrompt(new PromptInput(PromptTask.TRANSLATE_TO_GERMAN, "hi", "Antworte wie ein Pirat."));

        assertTrue(prompt.contains("<|im_start|>system\nAntworte wie ein Pirat.<|im_end|>\n"));
        assertFalse(prompt.contains("Translate the user's text"));
    }

    @Test
    void alreadyRenderedChatMlIsPassedThrough() {
        String rendered = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";

        String prompt = PromptStrategies.forModel("HuggingFaceTB/SmolLM2-135M-Instruct")
                .renderPrompt(PromptInput.of(PromptTask.TRANSLATE_TO_GERMAN, rendered));

        assertEquals(rendered, prompt);
    }


    // ---- Gemma 3 ----------------------------------------------------------

    @Test
    void gemma3InstructionModelRendersGemmaTurnTemplate() {
        String prompt = PromptStrategies.forModel("google/gemma-3-270m-it")
                .renderPrompt(PromptInput.of(PromptTask.SUMMARIZE, "Long text"));

        assertTrue(prompt.startsWith("<start_of_turn>user\nLong text\n\nSummarize"));
        assertTrue(prompt.contains("<end_of_turn>\n<start_of_turn>model\n"));
        assertFalse(prompt.contains("<|im_start|>"));
    }

    @Test
    void gemma3SupportsInteractiveWorkbenchTasks() {
        List<PromptTask> tasks = PromptStrategies.supportedTasks("google/gemma-3-270m-it");

        assertTrue(tasks.contains(PromptTask.NONE));
        assertTrue(tasks.contains(PromptTask.SUMMARIZE));
        assertTrue(tasks.contains(PromptTask.TRANSLATE_TO_GERMAN));
        assertTrue(tasks.contains(PromptTask.EXPLAIN_CODE));
    }

    // ---- T5 family --------------------------------------------------------

    @Test
    void googleT5TranslationUsesCanonicalPrefix() {
        String prompt = PromptStrategies.forModel("google-t5/t5-small")
                .renderPrompt(PromptInput.of(PromptTask.TRANSLATE_TO_GERMAN, "hello"));

        assertEquals("translate English to German: hello", prompt);
    }

    @Test
    void t5NonePassesTextThrough() {
        String prompt = PromptStrategies.forModel("google-t5/t5-small")
                .renderPrompt(PromptInput.of(PromptTask.NONE, "hello"));

        assertEquals("hello", prompt);
    }

    @Test
    void codeT5SmallOffersOnlyExplainCode() {
        List<PromptTask> tasks = PromptStrategies.supportedTasks("Salesforce/codet5-small");

        assertTrue(tasks.contains(PromptTask.NONE));
        assertTrue(tasks.contains(PromptTask.EXPLAIN_CODE));
        assertFalse(tasks.contains(PromptTask.TRANSLATE_TO_GERMAN));

        String prompt = PromptStrategies.forModel("Salesforce/codet5-small")
                .renderPrompt(PromptInput.of(PromptTask.EXPLAIN_CODE, "int x = 1;"));
        assertEquals("explain java: int x = 1;", prompt);
    }

    @Test
    void codeT5MultiSumOffersOnlySummary() {
        List<PromptTask> tasks = PromptStrategies.supportedTasks("Salesforce/codet5-base-multi-sum");

        assertTrue(tasks.contains(PromptTask.NONE));
        assertTrue(tasks.contains(PromptTask.SUMMARIZE));
        assertFalse(tasks.contains(PromptTask.TRANSLATE_TO_GERMAN));
        assertFalse(tasks.contains(PromptTask.EXPLAIN_CODE));
    }

    // ---- Qwen -------------------------------------------------------------

    @Test
    void qwenCoderSupportsExplainCodeAsChatMl() {
        List<PromptTask> tasks = PromptStrategies.supportedTasks("Qwen/Qwen2.5-Coder-0.5B-Instruct");
        assertTrue(tasks.contains(PromptTask.EXPLAIN_CODE));

        String prompt = PromptStrategies.forModel("Qwen/Qwen2.5-Coder-0.5B-Instruct")
                .renderPrompt(PromptInput.of(PromptTask.EXPLAIN_CODE, "class A {}"));
        assertFalse(prompt.contains("<|im_start|>system"));
        assertTrue(prompt.contains("<|im_start|>user\nclass A {}\n\nExplain the user's code"));
    }

    // ---- Phi-3 ------------------------------------------------------------

    @Test
    void phi3RendersPhi3ChatTemplateWithoutDefaultSystemPrompt() {
        String prompt = PromptStrategies.forModel("phi-3-mini-4k-instruct")
                .renderPrompt(PromptInput.of(PromptTask.NONE, "Hello"));

        assertFalse(prompt.contains("<|system|>"));
        assertTrue(prompt.contains("<|user|>\nHello<|end|>\n"));
        assertTrue(prompt.endsWith("<|assistant|>\n"));
    }

    // ---- Unknown / fallback ----------------------------------------------

    @Test
    void unknownModelFallsBackToRawPassThrough() {
        PromptStrategy strategy = PromptStrategies.forModel("some/unknown-model");

        assertEquals(List.of(PromptTask.NONE), PromptStrategies.supportedTasks("some/unknown-model"));
        assertEquals("verbatim", strategy.renderPrompt(PromptInput.of(PromptTask.TRANSLATE_TO_GERMAN, "verbatim")));
    }
}
