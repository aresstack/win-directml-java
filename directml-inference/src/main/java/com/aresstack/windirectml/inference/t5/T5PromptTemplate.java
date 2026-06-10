package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.InferenceRequest;

import java.util.Locale;
import java.util.Objects;

/**
 * Formats the optional instruction boundary for T5-family text-to-text models.
 */
final class T5PromptTemplate {
    private T5PromptTemplate() {
    }

    static String format(InferenceRequest request) {
        Objects.requireNonNull(request, "request");
        String text = request.getUserPrompt() == null ? "" : request.getUserPrompt().trim();
        if (text.isEmpty()) {
            return text;
        }
        String systemPrompt = request.getSystemPrompt() == null ? "" : request.getSystemPrompt().trim();
        if (systemPrompt.isEmpty() || hasKnownTaskPrefix(text)) {
            return text;
        }
        return formatSystemPrompt(systemPrompt, text);
    }

    private static String formatSystemPrompt(String systemPrompt, String text) {
        String normalized = systemPrompt.toLowerCase(Locale.ROOT).trim();
        if (normalized.equals("summarize") || normalized.equals("summary")
                || normalized.contains("fasse") || normalized.contains("zusammen")) {
            return "summarize: " + text;
        }
        if (normalized.contains("translate") || normalized.contains("übersetze") || normalized.contains("uebersetze")) {
            return systemPrompt + ": " + text;
        }
        if (normalized.contains("explain") || normalized.contains("erkläre") || normalized.contains("erklaere")) {
            return "explain: " + text;
        }
        return systemPrompt + "\n\n" + text;
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
