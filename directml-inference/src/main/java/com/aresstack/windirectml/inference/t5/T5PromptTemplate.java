package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.InferenceRequest;

import java.util.Locale;
import java.util.Objects;

/**
 * Formats workbench text into the narrow text-to-text prompts expected by T5 models.
 */
final class T5PromptTemplate {
    private T5PromptTemplate() {
    }

    static String format(InferenceRequest request) {
        Objects.requireNonNull(request, "request");
        String modelId = request.getModelId() == null ? "" : request.getModelId().toLowerCase(Locale.ROOT);
        String text = request.getUserPrompt() == null ? "" : request.getUserPrompt().trim();
        if (text.isEmpty()) {
            return text;
        }
        if (hasKnownTaskPrefix(text)) {
            return text;
        }
        if (modelId.contains("flan-t5") || modelId.contains("google-t5")
                || modelId.contains("/t5-small") || modelId.equals("t5-small")) {
            return "summarize: " + text;
        }
        if (modelId.contains("codet5")) {
            return "summarize java: " + text;
        }
        return text;
    }

    private static boolean hasKnownTaskPrefix(String text) {
        String lower = text.toLowerCase(Locale.ROOT);
        return lower.startsWith("summarize:")
                || lower.startsWith("translate ")
                || lower.startsWith("explain:")
                || lower.startsWith("explain java:")
                || lower.startsWith("summarize java:")
                || lower.startsWith("classify:");
    }
}
