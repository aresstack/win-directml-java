package com.aresstack.windirectml.inference;

import java.util.Objects;

/**
 * Request to a {@link Summarizer}.
 *
 * @param text         text to summarize (non-null, non-blank).
 * @param maxTokens    upper bound on generated tokens; {@code <= 0} = backend default.
 * @param systemPrompt optional system prompt; {@code null} = backend default.
 */
public record SummaryRequest(String text, int maxTokens, String systemPrompt) {

    public SummaryRequest {
        Objects.requireNonNull(text, "text");
        if (text.isBlank()) {
            throw new IllegalArgumentException("text must not be blank");
        }
    }

    public static SummaryRequest of(String text) {
        return new SummaryRequest(text, 0, null);
    }

    public static SummaryRequest of(String text, int maxTokens) {
        return new SummaryRequest(text, maxTokens, null);
    }
}

