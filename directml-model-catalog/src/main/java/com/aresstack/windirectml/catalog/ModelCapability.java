package com.aresstack.windirectml.catalog;

import java.util.Locale;

/**
 * A machine-readable model capability. These map 1:1 to the Ollama-compatible endpoints a local model can
 * serve and to the capability tokens {@code /api/show} reports, so a host can route strictly by capability
 * (never by guessing from the model name).
 *
 * <p>Java-8 compatible.</p>
 */
public enum ModelCapability {

    /** Produces embedding vectors ({@code /api/embed}, {@code /api/embeddings}). */
    EMBEDDING,
    /** Scores query/document pairs ({@code /api/rerank}). */
    RERANK,
    /** Free-form text completion ({@code /api/generate}). */
    COMPLETION,
    /** Instruction/chat turns with a chat template ({@code /api/chat}). */
    CHAT,
    /** Encoder-decoder sequence-to-sequence generation ({@code /api/generate}). */
    SEQ2SEQ,
    /** Summarization use case (a refinement of generation; never a standalone chat model unless CHAT is also set). */
    SUMMARIZE;

    /** The lowercase token used on the wire and in {@code /api/show} capability lists. */
    public String token() {
        return name().toLowerCase(Locale.ROOT);
    }
}
