package com.aresstack.windirectml.inference.prompt;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GENERATION-STREAMING-1 (prompt check): the Workbench Gemma prompt really lands in the Gemma chat
 * template with the user's text — not just a generic instruction. {@code PromptStrategies.forModel} for a
 * Gemma id returns {@link Gemma3PromptStrategy}, which emits the same {@code <start_of_turn>user … model}
 * turn markers as {@code Gemma3ChatTemplate}.
 */
class Gemma3PromptRoutingTest {

    @Test
    void noneTaskWrapsUserTextInChatTurnMarkers() {
        String text = "The capital of France is";
        String prompt = PromptStrategies.forModel("google/gemma-3-270m-it")
                .renderPrompt(PromptInput.of(PromptTask.NONE, text));

        assertTrue(prompt.contains("<start_of_turn>user"), "must open a user turn: " + prompt);
        assertTrue(prompt.contains(text), "must include the user's text: " + prompt);
        assertTrue(prompt.contains("<start_of_turn>model"), "must open the model turn: " + prompt);
        assertTrue(prompt.indexOf(text) < prompt.lastIndexOf("<start_of_turn>model"),
                "user text must precede the model turn: " + prompt);
    }

    @Test
    void alreadyRenderedChatPromptIsNotDoubleWrapped() {
        String rendered = "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n";
        String prompt = PromptStrategies.forModel("google/gemma-3-270m-it")
                .renderPrompt(PromptInput.of(PromptTask.NONE, rendered));
        // No second user turn appended.
        assertTrue(prompt.equals(rendered), "rendered chat prompt should pass through unchanged: " + prompt);
    }
}
