package com.aresstack.windirectml.config.generation;

/**
 * Shared result of a text-generation call.
 *
 * <p>All generation backends produce this type regardless of the
 * underlying model family (causal LM, seq2seq, summarizer adapter).
 *
 * <p>Java-8 compatible.
 */
public final class GenerationResult {

    private final String text;
    private final String finishReason;
    private final int promptTokens;
    private final int completionTokens;
    private final long elapsedMs;

    public GenerationResult(String text, String finishReason,
                            int promptTokens, int completionTokens, long elapsedMs) {
        this.text = text != null ? text : "";
        this.finishReason = finishReason != null ? finishReason : "unknown";
        this.promptTokens = promptTokens;
        this.completionTokens = completionTokens;
        this.elapsedMs = elapsedMs;
    }

    /**
     * Generated text output. Never null.
     */
    public String text() {
        return text;
    }

    /**
     * Finish reason: {@code "end_turn"}, {@code "max_tokens"}, {@code "stop_string"}, etc.
     */
    public String finishReason() {
        return finishReason;
    }

    /**
     * Number of tokens in the prompt.
     */
    public int promptTokens() {
        return promptTokens;
    }

    /**
     * Number of tokens generated.
     */
    public int completionTokens() {
        return completionTokens;
    }

    /**
     * Total tokens (prompt + completion).
     */
    public int totalTokens() {
        return promptTokens + completionTokens;
    }

    /**
     * Wall-clock time in milliseconds.
     */
    public long elapsedMs() {
        return elapsedMs;
    }

    @Override
    public String toString() {
        return "GenerationResult{finishReason='" + finishReason +
                "', completionTokens=" + completionTokens +
                ", elapsedMs=" + elapsedMs + "}";
    }
}
