package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.InferenceRequest;

import java.util.Locale;
import java.util.Objects;

/**
 * Formats the optional instruction boundary for T5-family text-to-text models.
 * <p>
 * Base T5 checkpoints are not chat models. They react best to canonical task
 * prefixes such as {@code summarize:} or {@code translate English to German:}.
 * Natural-language workbench instructions are therefore normalized before they
 * reach the tokenizer.
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
        String normalized = normalize(systemPrompt);
        if (isSummarizePrompt(normalized)) {
            return "summarize: " + text;
        }
        if (isTranslateToGermanPrompt(normalized)) {
            return "translate English to German: " + text;
        }
        if (isTranslateToEnglishPrompt(normalized)) {
            return "translate German to English: " + text;
        }
        if (isGenericTranslatePrompt(normalized)) {
            return "translate English to German: " + text;
        }
        if (isExplainPrompt(normalized)) {
            return "explain: " + text;
        }
        return systemPrompt + "\n\n" + text;
    }

    private static boolean isSummarizePrompt(String normalized) {
        return normalized.equals("summarize")
                || normalized.equals("summary")
                || normalized.contains("fasse")
                || normalized.contains("zusammen")
                || normalized.contains("summarize")
                || normalized.contains("summary");
    }

    private static boolean isTranslateToGermanPrompt(String normalized) {
        return normalized.equals("translate english to german")
                || normalized.contains("translate english to german")
                || normalized.contains("ins deutsche")
                || normalized.contains("into german")
                || normalized.contains("to german")
                || normalized.contains("nach deutsch");
    }

    private static boolean isTranslateToEnglishPrompt(String normalized) {
        return normalized.equals("translate german to english")
                || normalized.contains("translate german to english")
                || normalized.contains("ins englische")
                || normalized.contains("into english")
                || normalized.contains("to english")
                || normalized.contains("nach englisch");
    }

    private static boolean isGenericTranslatePrompt(String normalized) {
        return normalized.contains("translate")
                || normalized.contains("ubersetze")
                || normalized.contains("uebersetze");
    }

    private static boolean isExplainPrompt(String normalized) {
        return normalized.contains("explain")
                || normalized.contains("erklaere")
                || normalized.contains("erklar");
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

    private static String normalize(String value) {
        return value.toLowerCase(Locale.ROOT)
                .replace('ä', 'a')
                .replace('ö', 'o')
                .replace('ü', 'u')
                .replace('ß', 's')
                .trim();
    }
}
