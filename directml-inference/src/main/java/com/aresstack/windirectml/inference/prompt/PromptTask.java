package com.aresstack.windirectml.inference.prompt;

/**
 * Model-neutral user intent for a single generation run.
 *
 * <p>A task expresses <em>what</em> the caller wants ("translate to German"),
 * never <em>how</em> a concrete model is prompted. The translation from a task
 * to a model-correct prompt (ChatML system turn, T5 task prefix, ...) is owned
 * by the model-family specific {@link PromptStrategy} in the inference layer.</p>
 *
 * <p>This is the shared vocabulary between the (test-only) Workbench client and
 * the inference engines. The Workbench never builds prompt strings itself.</p>
 */
public enum PromptTask {

    /** Pass the user text through unchanged; no task instruction is added. */
    NONE,

    /** Produce a concise summary of the user text. */
    SUMMARIZE,

    /** Translate the user text into German. */
    TRANSLATE_TO_GERMAN,

    /** Translate the user text into English. */
    TRANSLATE_TO_ENGLISH,

    /** Explain the user-supplied code. */
    EXPLAIN_CODE
}
