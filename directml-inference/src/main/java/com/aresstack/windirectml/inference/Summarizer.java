package com.aresstack.windirectml.inference;

/**
 * Use-case API for text summarization.
 * <p>
 * Decouples the sidecar protocol from any particular inference backend.
 * Implementations may delegate to Phi-3 today, to other decoder LLMs later.
 */
public interface Summarizer {

    /**
     * @return {@code true} once the underlying model is loaded and ready
     *         to serve summarization requests.
     */
    boolean isReady();

    Summary summarize(SummaryRequest request) throws InferenceException;
}

