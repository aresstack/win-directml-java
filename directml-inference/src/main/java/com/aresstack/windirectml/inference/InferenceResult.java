package com.aresstack.windirectml.inference;

/**
 * The result produced by the local inference engine.
 * <p>
 * <b>V1 scope:</b> For MNIST, {@code text} contains a human-readable
 * classification result (e.g. "MNIST prediction: digit 7 (logits: …)"),
 * not generated prose. The {@code finishReason} is always "end_turn"
 * and {@code usage} reports pixel-count as input "tokens".
 * <p>
 * Future text-generation models will populate these fields with their
 * natural semantics (generated text, token counts, stop reasons).
 */
public class InferenceResult {

    private final String text;
    private final String finishReason;   // "end_turn" | "max_tokens" | "error"
    private final Usage usage;           // optional

    public InferenceResult(String text, String finishReason, Usage usage) {
        this.text = text;
        this.finishReason = finishReason;
        this.usage = usage;
    }

    /**
     * Convenience constructor without usage.
     */
    public InferenceResult(String text, String finishReason) {
        this(text, finishReason, null);
    }

    public String getText() {
        return text;
    }

    public String getFinishReason() {
        return finishReason;
    }

    public Usage getUsage() {
        return usage;
    }

    /**
     * Token usage statistics.
     */
    public record Usage(int promptTokens, int completionTokens, int totalTokens) {
    }

    @Override
    public String toString() {
        return "InferenceResult{finishReason='" + finishReason +
                "', textLength=" + (text != null ? text.length() : 0) +
                (usage != null ? ", usage=" + usage : "") + "}";
    }
}
