package com.aresstack.windirectml.inference;

/**
 * Result of a {@link Summarizer#summarize(SummaryRequest)} call.
 *
 * @param text         generated summary text.
 * @param finishReason {@code "end_turn"}, {@code "max_tokens"} or {@code "error"}.
 * @param promptTokens number of tokens fed into the model.
 * @param outputTokens number of tokens generated.
 * @param elapsedMs    wall-clock time in milliseconds.
 */
public record Summary(
        String text,
        String finishReason,
        int promptTokens,
        int outputTokens,
        long elapsedMs
) {
}

