package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.InferenceRequest;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class T5PromptTemplateTest {
    @Test
    void keepsGoogleT5SmallInputRawByDefault() {
        InferenceRequest request = InferenceRequest.builder()
                .modelId("google-t5/t5-small")
                .userPrompt("public class Demo {}")
                .build();

        assertEquals("public class Demo {}", T5PromptTemplate.format(request));
    }

    @Test
    void keepsGoogleFlanT5SmallInputRawByDefault() {
        InferenceRequest request = InferenceRequest.builder()
                .modelId("google/flan-t5-small")
                .userPrompt("public class Demo {}")
                .build();

        assertEquals("public class Demo {}", T5PromptTemplate.format(request));
    }

    @Test
    void appliesSummarizePrefixFromSystemPrompt() {
        InferenceRequest request = InferenceRequest.builder()
                .modelId("google-t5/t5-small")
                .systemPrompt("summarize")
                .userPrompt("public class Demo {}")
                .build();

        assertEquals("summarize: public class Demo {}", T5PromptTemplate.format(request));
    }

    @Test
    void keepsCodeT5InputRawByDefault() {
        InferenceRequest request = InferenceRequest.builder()
                .modelId("Salesforce/codet5-small")
                .userPrompt("public int size() { return count; }")
                .maxTokens(32)
                .build();

        assertEquals("public int size() { return count; }", T5PromptTemplate.format(request));
    }

    @Test
    void keepsExistingTaskPrefix() {
        InferenceRequest request = InferenceRequest.builder()
                .modelId("google-t5/t5-small")
                .userPrompt("translate English to German: hello")
                .build();

        assertEquals("translate English to German: hello", T5PromptTemplate.format(request));
    }

    @Test
    void keepsExplicitCodeT5TaskPrefix() {
        InferenceRequest request = InferenceRequest.builder()
                .modelId("Salesforce/codet5-small")
                .userPrompt("explain java: public int size() { return count; }")
                .maxTokens(32)
                .build();

        assertEquals("explain java: public int size() { return count; }", T5PromptTemplate.format(request));
    }

    @Test
    void mapsWorkbenchTranslateGermanTemplateToCanonicalT5Prefix() {
        InferenceRequest request = InferenceRequest.builder()
                .modelId("google-t5/t5-small")
                .systemPrompt("translate English to German")
                .userPrompt("Paste a longer text or prompt here.")
                .build();

        assertEquals("translate English to German: Paste a longer text or prompt here.", T5PromptTemplate.format(request));
    }

    @Test
    void mapsGermanWorkbenchTranslateInstructionToCanonicalT5Prefix() {
        InferenceRequest request = InferenceRequest.builder()
                .modelId("google-t5/t5-small")
                .systemPrompt("Übersetze den folgenden Text ins Deutsche. Gib nur die Übersetzung aus.")
                .userPrompt("Paste a longer text or prompt here.")
                .build();

        assertEquals("translate English to German: Paste a longer text or prompt here.", T5PromptTemplate.format(request));
    }

    @Test
    void mapsWorkbenchTranslateEnglishTemplateToCanonicalT5Prefix() {
        InferenceRequest request = InferenceRequest.builder()
                .modelId("google-t5/t5-small")
                .systemPrompt("translate German to English")
                .userPrompt("Füge hier einen längeren Text ein.")
                .build();

        assertEquals("translate German to English: Füge hier einen längeren Text ein.", T5PromptTemplate.format(request));
    }

}
