package com.aresstack.windirectml.inference.qwen;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link QwenChatTemplate}.
 *
 * <p>Verifies that the ChatML format is deterministic and correct for:
 * <ul>
 *   <li>Single-turn summarization prompts (English)</li>
 *   <li>Single-turn summarization prompts (German)</li>
 *   <li>Code explanation prompts (Natural/ADABAS-like pseudo-code)</li>
 *   <li>Multi-turn conversations</li>
 * </ul>
 */
class QwenChatTemplateTest {

    // ── Single-turn format ───────────────────────────────────────────────

    @Test
    void singleTurnFormatWithSystemPrompt() {
        String result = QwenChatTemplate.formatChat(
                "You are a helpful assistant.",
                "What is Java?");

        assertEquals("""
                <|im_start|>system
                You are a helpful assistant.<|im_end|>
                <|im_start|>user
                What is Java?<|im_end|>
                <|im_start|>assistant
                """, result);
    }

    @Test
    void singleTurnFormatWithoutSystemPrompt() {
        String result = QwenChatTemplate.formatChat(null, "Hello");

        assertEquals("""
                <|im_start|>system
                You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
                <|im_start|>user
                Hello<|im_end|>
                <|im_start|>assistant
                """, result);
    }

    @Test
    void singleTurnFormatWithEmptySystemPrompt() {
        String result = QwenChatTemplate.formatChat("", "Hello");

        assertEquals("""
                <|im_start|>system
                You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
                <|im_start|>user
                Hello<|im_end|>
                <|im_start|>assistant
                """, result);
    }

    @Test
    void defaultSystemPromptMatchesQwenInstructTemplate() {
        assertEquals(
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                QwenChatTemplate.DEFAULT_SYSTEM_PROMPT);
    }

    @Test
    void singleTurnFormatWithoutSystemPromptExplicitOptOut() {
        String result = QwenChatTemplate.formatChat(null, "Hello", false);

        assertEquals("""
                <|im_start|>user
                Hello<|im_end|>
                <|im_start|>assistant
                """, result);
    }

    // ── Summarization prompt (English) ───────────────────────────────────

    @Test
    void summarizationPromptEnglish() {
        String systemPrompt = "You are a summarization assistant. Provide concise summaries.";
        String userMessage = "Summarize the following paragraph:\n\n" +
                "Java is a high-level, class-based, object-oriented programming language " +
                "that is designed to have as few implementation dependencies as possible.";

        String result = QwenChatTemplate.formatChat(systemPrompt, userMessage);

        assertTrue(result.startsWith("<|im_start|>system\n"));
        assertTrue(result.contains("summarization assistant"));
        assertTrue(result.contains("<|im_end|>\n<|im_start|>user\n"));
        assertTrue(result.contains("Summarize the following paragraph"));
        assertTrue(result.endsWith("<|im_start|>assistant\n"));
    }

    // ── Summarization prompt (German) ────────────────────────────────────

    @Test
    void summarizationPromptGerman() {
        String systemPrompt = "Du bist ein hilfreicher Assistent für Zusammenfassungen.";
        String userMessage = "Fasse den folgenden Absatz zusammen:\n\n" +
                "Java ist eine objektorientierte Programmiersprache, die einen " +
                "besonderen Schwerpunkt auf Plattformunabhängigkeit legt.";

        String result = QwenChatTemplate.formatChat(systemPrompt, userMessage);

        assertTrue(result.contains("hilfreicher Assistent"));
        assertTrue(result.contains("Fasse den folgenden Absatz"));
        assertTrue(result.contains("Plattformunabhängigkeit"));
        assertEquals("""
                <|im_start|>system
                Du bist ein hilfreicher Assistent für Zusammenfassungen.<|im_end|>
                <|im_start|>user
                Fasse den folgenden Absatz zusammen:
                
                Java ist eine objektorientierte Programmiersprache, die einen besonderen Schwerpunkt auf Plattformunabhängigkeit legt.<|im_end|>
                <|im_start|>assistant
                """, result);
    }

    // ── Code explanation prompt (Natural/ADABAS-like) ─────────────────────

    @Test
    void codeExplanationPromptNaturalAdabas() {
        String systemPrompt = "You are a code explanation assistant. Explain the following code snippet.";
        String codeSnippet = """
                Explain the following Natural/ADABAS pseudo-code:
                
                DEFINE DATA LOCAL
                  1 #EMPLOYEE-NAME (A30)
                  1 #SALARY (N7.2)
                END-DEFINE
                
                READ EMPLOYEES BY NAME
                  DISPLAY #EMPLOYEE-NAME #SALARY
                END-READ
                
                END""";

        String result = QwenChatTemplate.formatChat(systemPrompt, codeSnippet);

        assertTrue(result.contains("DEFINE DATA LOCAL"));
        assertTrue(result.contains("#EMPLOYEE-NAME"));
        assertTrue(result.contains("READ EMPLOYEES BY NAME"));
        assertTrue(result.endsWith("<|im_start|>assistant\n"));
    }

    // ── Multi-turn conversation ──────────────────────────────────────────

    @Test
    void multiTurnConversation() {
        String result = QwenChatTemplate.formatMultiTurnChat(
                "You are a coding assistant.",
                List.of(
                        QwenChatTemplate.ChatMessage.user("What is BPE?"),
                        QwenChatTemplate.ChatMessage.assistant("BPE is Byte Pair Encoding, a tokenization algorithm."),
                        QwenChatTemplate.ChatMessage.user("How does it work?")
                ));

        assertEquals("""
                <|im_start|>system
                You are a coding assistant.<|im_end|>
                <|im_start|>user
                What is BPE?<|im_end|>
                <|im_start|>assistant
                BPE is Byte Pair Encoding, a tokenization algorithm.<|im_end|>
                <|im_start|>user
                How does it work?<|im_end|>
                <|im_start|>assistant
                """, result);
    }

    // ── Determinism ──────────────────────────────────────────────────────

    @Test
    void formatIsDeterministic() {
        String a = QwenChatTemplate.formatChat("System", "User message");
        String b = QwenChatTemplate.formatChat("System", "User message");
        assertEquals(a, b, "Chat template output must be deterministic");
    }
}
