package com.aresstack.windirectml.workbench.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.aresstack.windirectml.inference.prompt.ChatTaskInstructions;
import com.aresstack.windirectml.inference.prompt.PromptInput;
import com.aresstack.windirectml.inference.prompt.PromptStrategies;
import com.aresstack.windirectml.inference.prompt.PromptTask;
import java.lang.reflect.Method;
import java.util.Locale;
import org.junit.jupiter.api.Test;

/**
 * P7 — the workbench applies the selected {@link PromptTask} by delegating to the shared family-aware
 * {@link PromptStrategies} pipeline inside {@link WorkbenchGenerationService}, then hands the rendered
 * text to the runtime as {@code userPrompt}. The runtime renders once more with {@link PromptTask#NONE};
 * because every strategy is idempotent, the family template / task prefix is applied <b>exactly once</b>.
 *
 * <p>Each test asserts (a) the family-correct rendering for the task and (b) idempotency under a second
 * {@code NONE} render — the exact two-stage contract between the workbench and the neutral runtime.</p>
 */
class WorkbenchPromptRenderingTest {

    private static final String T5 = "google-t5/t5-small";
    private static final String CHATML = "Qwen/Qwen2.5-Coder-0.5B-Instruct";
    private static final String GEMMA = "google/gemma-3-270m-it";
    private static final String PHI3 = "microsoft/Phi-3-mini-4k-instruct-onnx";

    private static int count(String haystack, String needle) {
        int n = 0;
        for (int i = haystack.indexOf(needle); i >= 0; i = haystack.indexOf(needle, i + needle.length())) {
            n++;
        }
        return n;
    }

    /** The runtime's second render pass: model-keyed strategy with PromptTask.NONE over the rendered text. */
    private static String runtimeRerender(String modelId, String rendered) {
        return PromptStrategies.forModel(modelId).renderPrompt(PromptInput.of(PromptTask.NONE, rendered));
    }

    @Test
    void t5SummarizeHasExactlyOnePrefixAndIsIdempotent() {
        String stage1 = WorkbenchGenerationService.renderPrompt(T5, PromptTask.SUMMARIZE, "Ein langer Text.");
        assertTrue(stage1.toLowerCase(Locale.ROOT).startsWith("summarize: "), "T5 needs the summarize prefix: " + stage1);
        assertEquals(1, count(stage1.toLowerCase(Locale.ROOT), "summarize:"), "exactly one prefix");

        String stage2 = runtimeRerender(T5, stage1);
        assertEquals(stage1, stage2, "runtime re-render must not add a second prefix");
        assertEquals(1, count(stage2.toLowerCase(Locale.ROOT), "summarize:"), "still exactly one prefix");
    }

    @Test
    void chatMlTranslateHasExactlyOneTemplateWithInstructionInUserTurn() {
        String instruction = ChatTaskInstructions.standard().instructionFor(PromptTask.TRANSLATE_TO_GERMAN);
        String stage1 = WorkbenchGenerationService.renderPrompt(CHATML, PromptTask.TRANSLATE_TO_GERMAN, "Hello world.");

        assertEquals(1, count(stage1, "<|im_start|>assistant"), "exactly one ChatML assistant turn");
        assertEquals(1, count(stage1, instruction), "task instruction appears exactly once");
        // Instruction lives in the user turn (after the last user marker, before the assistant turn).
        int lastUser = stage1.lastIndexOf("<|im_start|>user");
        int assistant = stage1.indexOf("<|im_start|>assistant");
        int instrAt = stage1.indexOf(instruction);
        assertTrue(lastUser >= 0 && instrAt > lastUser && instrAt < assistant,
                "instruction must sit in the user turn: " + stage1);

        String stage2 = runtimeRerender(CHATML, stage1);
        assertEquals(stage1, stage2, "idempotent: no second ChatML template");
    }

    @Test
    void gemmaSummarizeHasExactlyOneTurnTemplateAndIsIdempotent() {
        String stage1 = WorkbenchGenerationService.renderPrompt(GEMMA, PromptTask.SUMMARIZE, "Ein Text.");
        assertEquals(2, count(stage1, "<start_of_turn>"), "one user + one model turn marker");
        assertEquals(1, count(stage1, "<end_of_turn>"), "one end-of-turn marker");

        String stage2 = runtimeRerender(GEMMA, stage1);
        assertEquals(stage1, stage2, "idempotent: no second Gemma turn template");
    }

    @Test
    void phi3TranslateHasExactlyOneTemplateWithInstructionInSystemTurn() {
        String instruction = ChatTaskInstructions.standard().instructionFor(PromptTask.TRANSLATE_TO_ENGLISH);
        String stage1 = WorkbenchGenerationService.renderPrompt(PHI3, PromptTask.TRANSLATE_TO_ENGLISH, "Hallo Welt.");

        assertEquals(1, count(stage1, "<|user|>"), "exactly one Phi-3 user turn");
        assertEquals(1, count(stage1, "<|assistant|>"), "exactly one Phi-3 assistant turn");
        assertEquals(1, count(stage1, instruction), "task instruction appears exactly once");
        // Instruction lives in the system turn (before the user turn).
        int system = stage1.indexOf("<|system|>");
        int user = stage1.indexOf("<|user|>");
        int instrAt = stage1.indexOf(instruction);
        assertTrue(system >= 0 && instrAt > system && instrAt < user,
                "instruction must sit in the Phi-3 system turn: " + stage1);

        String stage2 = runtimeRerender(PHI3, stage1);
        assertEquals(stage1, stage2, "idempotent: no second Phi-3 template");
    }

    @Test
    void noneAppliesNoTaskInstructionButKeepsTheFamilyTemplateExactlyOnce() {
        // Chat family: template present, but no task instruction text.
        String summarize = ChatTaskInstructions.standard().instructionFor(PromptTask.SUMMARIZE);
        String chat = WorkbenchGenerationService.renderPrompt(CHATML, PromptTask.NONE, "Plain text.");
        assertEquals(1, count(chat, "<|im_start|>assistant"), "family template still applied once");
        assertFalse(chat.contains(summarize), "NONE must not inject a task instruction");
        assertEquals(chat, runtimeRerender(CHATML, chat), "idempotent under NONE");

        // T5 family: NONE means no prefix at all, and re-render stays identical.
        String t5 = WorkbenchGenerationService.renderPrompt(T5, PromptTask.NONE, "Plain text.");
        assertEquals(0, count(t5.toLowerCase(Locale.ROOT), "summarize:"), "NONE adds no T5 prefix");
        assertEquals(t5, runtimeRerender(T5, t5), "idempotent under NONE");
    }

    @Test
    void serviceGenerateAcceptsThePromptTaskSelection() throws Exception {
        // The panel captures the selected PromptTask and hands it to the service (compile-verified at the
        // call site); assert the service signature actually carries a PromptTask so the task can flow.
        Method generate = WorkbenchGenerationService.class.getMethod("generate",
                String.class, java.nio.file.Path.class,
                com.aresstack.windirectml.catalog.CatalogBackend.class,
                PromptTask.class, String.class, int.class,
                com.aresstack.windirectml.inference.api.GenerationTokenListener.class);
        assertEquals(PromptTask.class, generate.getParameterTypes()[3],
                "WorkbenchGenerationService.generate must accept the selected PromptTask");
    }
}
