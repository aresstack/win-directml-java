package com.aresstack.windirectml.sidecar.client;

import com.fasterxml.jackson.databind.JsonNode;

/**
 * Typed view on the {@code summarize} response.
 */
public final class SummaryResult {

    private final String text;
    private final String finishReason;
    private final int promptTokens;
    private final int outputTokens;
    private final long elapsedMillis;
    private final String raw;

    public SummaryResult(String text, String finishReason, int promptTokens,
                         int outputTokens, long elapsedMillis, String raw) {
        this.text = text;
        this.finishReason = finishReason;
        this.promptTokens = promptTokens;
        this.outputTokens = outputTokens;
        this.elapsedMillis = elapsedMillis;
        this.raw = raw;
    }

    public static SummaryResult from(JsonNode result, long elapsedMillis, String raw) {
        if (result == null) {
            return new SummaryResult(null, null, 0, 0, elapsedMillis, raw);
        }
        String text = result.has("text") && !result.get("text").isNull()
                ? result.get("text").asText() : null;
        String finishReason = result.has("finishReason") && !result.get("finishReason").isNull()
                ? result.get("finishReason").asText() : null;
        int prompt = result.has("promptTokens") ? result.get("promptTokens").asInt(0) : 0;
        int output = result.has("outputTokens") ? result.get("outputTokens").asInt(0) : 0;
        long sidecarElapsed = result.has("elapsedMs") ? result.get("elapsedMs").asLong(0L) : elapsedMillis;
        return new SummaryResult(text, finishReason, prompt, output, sidecarElapsed, raw);
    }

    public String getText() {
        return text;
    }

    public String getFinishReason() {
        return finishReason;
    }

    public int getPromptTokens() {
        return promptTokens;
    }

    public int getOutputTokens() {
        return outputTokens;
    }

    public long getElapsedMillis() {
        return elapsedMillis;
    }

    public String getRaw() {
        return raw;
    }
}

