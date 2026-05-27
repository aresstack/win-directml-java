package com.aresstack.windirectml.inference.qwen;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Optional real-model smoke test for Qwen2.5-Coder 0.5B CPU generation.
 *
 * <p><b>These tests require real model weights and are skipped by default in CI.</b>
 * To run locally, download the Qwen2.5-Coder-0.5B-Instruct ONNX model into
 * {@code model/qwen2.5-coder-0.5b-instruct/} (or set system property
 * {@code qwen.model.dir} to point at the model directory).</p>
 *
 * <h3>Manual run command:</h3>
 * <pre>{@code
 * ./gradlew :directml-inference:test \
 *     --tests "*.qwen.QwenCpuSmokeTest" \
 *     -Dqwen.model.dir=model/qwen2.5-coder-0.5b-instruct
 * }</pre>
 *
 * <h3>Prompts tested:</h3>
 * <ul>
 *   <li><b>English summarization</b> — summarize a short English text</li>
 *   <li><b>German summarization</b> — summarize a short German text</li>
 *   <li><b>Natural/ADABAS code explanation</b> — explain a code snippet</li>
 *   <li><b>Short max-token generation</b> — verify that output is produced
 *       with a small max-token limit</li>
 * </ul>
 *
 * <p>This test validates the CPU runtime path from issue #99 and the model
 * download layout from issue #100. Tokenizer/template tests from issue #98
 * cover the ChatML formatting separately.</p>
 *
 * @see QwenModelDirValidator
 */
@EnabledIf("modelPresent")
class QwenCpuSmokeTest {

    // ── Prompt definitions ───────────────────────────────────────────────

    /** English summarization prompt. */
    static final String ENGLISH_SUMMARY_PROMPT =
            "Summarize the following text in one sentence:\n\n"
            + "The Qwen2.5-Coder series is a code-specific large language model built "
            + "upon the Qwen2.5 architecture. It has been pretrained on a large corpus "
            + "of source code and demonstrates strong coding ability. The 0.5B variant "
            + "is the smallest model in the series and is suitable for lightweight, "
            + "low-latency code generation tasks on CPU.";

    /** German summarization prompt. */
    static final String GERMAN_SUMMARY_PROMPT =
            "Fasse den folgenden Text in einem Satz zusammen:\n\n"
            + "Die Qwen2.5-Coder-Reihe ist eine auf Code spezialisierte Familie großer "
            + "Sprachmodelle, die auf der Qwen2.5-Architektur basiert. Sie wurde auf einem "
            + "großen Korpus von Quellcode vortrainiert und zeigt starke Programmierfähigkeiten. "
            + "Die 0.5B-Variante ist das kleinste Modell der Serie und eignet sich für "
            + "ressourcenschonende Code-Generierung auf der CPU.";

    /** Natural/ADABAS code explanation prompt. */
    static final String NATURAL_ADABAS_CODE_EXPLANATION_PROMPT =
            "Explain what the following Natural/ADABAS code does:\n\n"
            + "DEFINE DATA LOCAL\n"
            + "1 #EMPLOYEES VIEW OF EMPLOYEES\n"
            + "  2 NAME\n"
            + "  2 FIRST-NAME\n"
            + "  2 PERSONNEL-ID\n"
            + "END-DEFINE\n"
            + "READ #EMPLOYEES BY NAME\n"
            + "  DISPLAY NAME FIRST-NAME PERSONNEL-ID\n"
            + "END-READ\n"
            + "END";

    /** System prompt for code assistant role. */
    static final String SYSTEM_PROMPT = "You are a helpful coding assistant.";

    /** Short max-token limit for verifying basic generation works. */
    static final int SHORT_MAX_TOKENS = 32;

    // ── Condition ────────────────────────────────────────────────────────

    static boolean modelPresent() {
        Path dir = resolveModelDir();
        return dir != null && QwenModelDirValidator.isValidModelDir(dir);
    }

    private static Path resolveModelDir() {
        String explicit = System.getProperty("qwen.model.dir");
        if (explicit != null && !explicit.isBlank()) {
            return Path.of(explicit);
        }
        // Default location relative to repository root
        Path defaultDir = Path.of("model", "qwen2.5-coder-0.5b-instruct");
        if (Files.isDirectory(defaultDir)) {
            return defaultDir;
        }
        return null;
    }

    // ── Smoke tests ──────────────────────────────────────────────────────

    @Test
    @Disabled("Awaiting Qwen CPU runtime (#99)")
    void englishSummarizationProducesOutput() {
        // This test verifies that the CPU runtime generates non-empty output
        // for an English summarization prompt.
        //
        // Implementation note: once the Qwen runtime (issue #99) lands, this
        // test should instantiate the runtime, format the prompt using ChatML
        // (issue #98), generate with SHORT_MAX_TOKENS, and assert non-empty.
        //
        // Placeholder assertion that validates model directory:
        Path dir = resolveModelDir();
        assertNotNull(dir, "Model directory must be resolved when modelPresent() is true");
        assertTrue(QwenModelDirValidator.isValidModelDir(dir));

        // TODO(#99): Replace with actual Qwen runtime generation call:
        // var engine = new QwenInferenceEngine(dir, SHORT_MAX_TOKENS, "cpu");
        // engine.initialize();
        // var result = engine.generate(new InferenceRequest(SYSTEM_PROMPT, ENGLISH_SUMMARY_PROMPT, SHORT_MAX_TOKENS));
        // assertFalse(result.getText().isBlank(), "Generated text must not be empty");
        // engine.shutdown();
    }

    @Test
    @Disabled("Awaiting Qwen CPU runtime (#99)")
    void germanSummarizationProducesOutput() {
        Path dir = resolveModelDir();
        assertNotNull(dir);
        assertTrue(QwenModelDirValidator.isValidModelDir(dir));

        // TODO(#99): Replace with actual Qwen runtime generation call:
        // var engine = new QwenInferenceEngine(dir, SHORT_MAX_TOKENS, "cpu");
        // engine.initialize();
        // var result = engine.generate(new InferenceRequest(SYSTEM_PROMPT, GERMAN_SUMMARY_PROMPT, SHORT_MAX_TOKENS));
        // assertFalse(result.getText().isBlank(), "German summary must not be empty");
        // engine.shutdown();
    }

    @Test
    @Disabled("Awaiting Qwen CPU runtime (#99)")
    void naturalAdabasCodeExplanationProducesOutput() {
        Path dir = resolveModelDir();
        assertNotNull(dir);
        assertTrue(QwenModelDirValidator.isValidModelDir(dir));

        // TODO(#99): Replace with actual Qwen runtime generation call:
        // var engine = new QwenInferenceEngine(dir, 64, "cpu");
        // engine.initialize();
        // var result = engine.generate(new InferenceRequest(SYSTEM_PROMPT, NATURAL_ADABAS_CODE_EXPLANATION_PROMPT, 64));
        // assertFalse(result.getText().isBlank(), "Code explanation must not be empty");
        // engine.shutdown();
    }

    @Test
    @Disabled("Awaiting Qwen CPU runtime (#99)")
    void shortMaxTokenGenerationProducesOutput() {
        Path dir = resolveModelDir();
        assertNotNull(dir);
        assertTrue(QwenModelDirValidator.isValidModelDir(dir));

        // TODO(#99): Replace with actual Qwen runtime generation call:
        // var engine = new QwenInferenceEngine(dir, SHORT_MAX_TOKENS, "cpu");
        // engine.initialize();
        // var result = engine.generate(new InferenceRequest(SYSTEM_PROMPT, "Hello!", SHORT_MAX_TOKENS));
        // assertFalse(result.getText().isBlank(), "Short generation must produce output");
        // assertTrue(result.getUsage().completionTokens() <= SHORT_MAX_TOKENS,
        //         "Must not exceed max token limit");
        // engine.shutdown();
    }
}
