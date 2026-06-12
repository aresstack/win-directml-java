package com.aresstack.windirectml.inference.qwen;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Slice 10 — verifies the opt-in {@code -Dqwen.runtime=decoderonly-session} routing of the real generation entry
 * point ({@link Qwen2Runtime#generateStreaming}).
 *
 * <p>Runs the SAME entry point twice on the SAME loaded model — once with the default (legacy) runtime and once with
 * the session runtime selected via the system property — and asserts identical streamed token ids, identical decoded
 * text and identical finish reason. The legacy path is the gold standard and is unchanged.</p>
 *
 * <p><b>Skipped by default.</b> Requires a real Qwen model + opt-in. Manual run:</p>
 * <pre>{@code
 * ./gradlew :directml-inference:test --tests "*.qwen.QwenSessionRoutingE2eTest" \
 *     -Dqwen.enable.experimental.runtime=true \
 *     -Dqwen.model.dir=<path-to>/qwen2.5-coder-0.5b-directml-int4 \
 *     -Dqwen.harness.modelfile=model_q4f16.onnx --info
 * }</pre>
 */
@EnabledIf("routingEnabled")
class QwenSessionRoutingE2eTest {

    private static final Logger log = LoggerFactory.getLogger(QwenSessionRoutingE2eTest.class);

    private static final int MAX_TOKENS = Integer.getInteger("qwen.harness.maxtokens", 24);
    private static final String BACKEND = System.getProperty("qwen.harness.backend", "auto");
    private static final String MODEL_FILE =
            System.getProperty("qwen.harness.modelfile", QwenModelDirValidator.DEFAULT_MODEL_FILE);
    private static final String PROMPT =
            "<|im_start|>user\nWrite a one-line Python function that adds two numbers.<|im_end|>\n"
                    + "<|im_start|>assistant\n";

    private static QwenInferenceEngine engine;
    private static Qwen2Runtime runtime;

    static boolean routingEnabled() {
        if (!"true".equalsIgnoreCase(System.getProperty("qwen.enable.experimental.runtime"))) {
            return false;
        }
        Path dir = resolveModelDir();
        return dir != null && QwenModelDirValidator.isValidModelDir(dir);
    }

    private static Path resolveModelDir() {
        String explicit = System.getProperty("qwen.model.dir");
        if (explicit != null && !explicit.isBlank()) {
            return Path.of(explicit);
        }
        Path defaultDir = Path.of("model", "qwen2.5-coder-0.5b-directml-int4");
        return Files.isDirectory(defaultDir) ? defaultDir : null;
    }

    @BeforeAll
    static void setUp() throws Exception {
        Path dir = resolveModelDir();
        assertNotNull(dir, "Model directory must resolve when routing test is enabled");
        engine = new QwenInferenceEngine(dir, MAX_TOKENS, BACKEND, MODEL_FILE);
        engine.initialize();
        assertTrue(engine.isReady(), "Engine must be ready");
        runtime = engine.getRuntime();
    }

    @AfterAll
    static void tearDown() {
        if (engine != null) {
            engine.shutdown();
        }
        System.clearProperty("qwen.runtime");
    }

    @Test
    void sessionRuntimeRoutingMatchesLegacyTokenForToken() {
        Run legacy;
        Run session;
        Run defaultRun;
        try {
            System.setProperty("qwen.runtime", "legacy"); // explicit legacy (no longer the default as of slice 11a)
            legacy = runOnce();
            System.setProperty("qwen.runtime", "decoderonly-session");
            session = runOnce();
            System.clearProperty("qwen.runtime"); // no property → must now default to the session path
            defaultRun = runOnce();
        } finally {
            System.clearProperty("qwen.runtime");
        }

        log.info("[routing] legacy : {} tokens, finish={}, ids={}", legacy.tokens.size(), legacy.finish, legacy.tokens);
        log.info("[routing] session: {} tokens, finish={}, ids={}", session.tokens.size(), session.finish, session.tokens);
        log.info("[routing] default: {} tokens, finish={}, ids={}",
                defaultRun.tokens.size(), defaultRun.finish, defaultRun.tokens);

        assertEquals(legacy.tokens, session.tokens, "Session routing must stream the same token ids as legacy");
        assertEquals(legacy.text, session.text, "Session routing must produce the same decoded text as legacy");
        assertEquals(legacy.finish, session.finish, "Finish reason must match legacy");

        // Slice 11a: without a property, generation must take the session path (== explicit session run).
        assertEquals(session.tokens, defaultRun.tokens, "Default (no property) must route to the session path");
        assertEquals(session.text, defaultRun.text, "Default (no property) must match the session path output");
    }

    private Run runOnce() {
        List<Integer> tokens = new ArrayList<>();
        String text = runtime.generateStreaming(PROMPT, MAX_TOKENS,
                (tokenId, fullText, delta) -> tokens.add(tokenId));
        String finish = tokens.size() < MAX_TOKENS ? "eos_token" : "length";
        return new Run(tokens, text, finish);
    }

    private record Run(List<Integer> tokens, String text, String finish) {
    }
}
