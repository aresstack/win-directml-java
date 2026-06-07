package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.InferenceRequest;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class T5PromptTemplateTest {
    @Test
    void addsSummarizePrefixForGoogleT5Small() {
        InferenceRequest request = InferenceRequest.builder()
                .modelId("google-t5/t5-small")
                .userPrompt("public class Demo {}")
                .build();

        assertEquals("summarize: public class Demo {}", T5PromptTemplate.format(request));
    }

    @Test
    void addsSummarizePrefixForGoogleFlanT5Small() {
        InferenceRequest request = InferenceRequest.builder()
                .modelId("google/flan-t5-small")
                .userPrompt("public class Demo {}")
                .build();

        assertEquals("summarize: public class Demo {}", T5PromptTemplate.format(request));
    }

    @Test
    void formatsCodeT5WithJavaSummarizationPrefix() {
        InferenceRequest request = InferenceRequest.builder()
                .modelId("Salesforce/codet5-small")
                .userPrompt("public int size() { return count; }")
                .maxTokens(32)
                .build();

        assertEquals("summarize java: public int size() { return count; }", T5PromptTemplate.format(request));
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
}
