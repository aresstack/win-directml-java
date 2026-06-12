package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.InferenceRequest;
import com.aresstack.windirectml.inference.InferenceResult;
import com.aresstack.windirectml.inference.prompt.PromptTask;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Real-model T5 end-to-end smoke (slice T5-2): drives a local google-t5/t5-small through the reference engine and
 * verifies the full path (tokenizer -> encoder -> decoder step loop -> cross-attention -> LM head -> token select ->
 * decode) produces a non-empty output. Enabled only when a local t5-small directory is present (override with
 * {@code -Dt5.testModelDir=...}); skipped in normal CI.
 */
@EnabledIf("modelPresent")
class T5RealModelReferenceTest {

    static Path resolveModelDir() {
        String override = System.getProperty("t5.testModelDir");
        if (override != null && !override.isBlank()) {
            Path p = Path.of(override);
            return Files.isRegularFile(p.resolve("config.json")) ? p : null;
        }
        for (Path p : new Path[]{Path.of("model/t5-small"), Path.of("../model/t5-small")}) {
            if (Files.isRegularFile(p.resolve("config.json"))) {
                return p;
            }
        }
        return null;
    }

    static boolean modelPresent() {
        return resolveModelDir() != null;
    }

    @Test
    void t5SmallReferenceProducesNonEmptySummary() throws Exception {
        Path modelDir = resolveModelDir();
        T5InferenceEngine engine = new T5InferenceEngine(modelDir, 20, "reference");
        try {
            engine.initialize();
            assertTrue(engine.isReady(), "T5 engine must initialise from the local t5-small package");

            InferenceResult result = engine.generate(InferenceRequest.builder()
                    .modelId("google-t5/t5-small")
                    .task(PromptTask.SUMMARIZE)
                    .userPrompt("The quick brown fox jumps over the lazy dog.")
                    .maxTokens(20)
                    .temperature(0.0f)
                    .build());

            System.out.println("[T5-E2E] executionMode=" + engine.executionMode());
            System.out.println("[T5-E2E] result=" + result);
            System.out.println("[T5-E2E] tokenPreview=" + engine.lastOutputTokenPreview());
            System.out.println("[T5-E2E] OUTPUT=[" + result.getText() + "]");

            assertNotNull(result.getText(), "T5 output text must not be null");
            assertTrue(result.getUsage().completionTokens() > 0, "T5 must generate at least one token");
            assertFalse(result.getText().isBlank(), "T5 output must not be empty/blank");
        } finally {
            engine.shutdown();
        }
    }
}
