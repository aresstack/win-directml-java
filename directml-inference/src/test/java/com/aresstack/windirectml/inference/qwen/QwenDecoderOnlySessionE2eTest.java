package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyForwardPass;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGenerationLoop;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGenerationResult;
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
 * Slice 7 — Qwen decoder-only session E2E verification.
 *
 * <p>Runs the experimental {@link QwenDecoderOnlyForwardPass}/{@link DecoderOnlyGenerationLoop} path against the
 * production {@link Qwen2Runtime} on the SAME loaded model, with the same prompt, maxTokens and greedy/penalty
 * configuration, and compares the produced token ids, finish reason and runtime. The production runtime is the gold
 * standard; this test never changes it.</p>
 *
 * <p><b>Skipped by default.</b> Requires a real Qwen model and explicit opt-in. Manual run:</p>
 * <pre>{@code
 * ./gradlew :directml-inference:test --tests "*.qwen.QwenDecoderOnlySessionE2eTest" \
 *     -Dqwen.enable.experimental.runtime=true \
 *     -Dqwen.decoderonly.experimental=true \
 *     -Dqwen.model.dir=<path-to>/qwen2.5-coder-0.5b-directml-int4 \
 *     --info
 * }</pre>
 *
 * <h3>Result contract (harmonised in slice 8)</h3>
 * <p>Both loops now use the same contract: a terminating stop token ends generation but is NOT recorded as a generated
 * or streamed token. The shared {@link DecoderOnlyGenerationLoop} reports it via {@code finishTokenId} instead. So the
 * generated token ids, the streamed token ids and the finish reason all match the production runtime directly, with no
 * stop-token special case.</p>
 */
@EnabledIf("harnessEnabled")
class QwenDecoderOnlySessionE2eTest {

    private static final Logger log = LoggerFactory.getLogger(QwenDecoderOnlySessionE2eTest.class);

    private static final int MAX_TOKENS = Integer.getInteger("qwen.harness.maxtokens", 16);
    private static final float PENALTY =
            Float.parseFloat(System.getProperty("qwen.harness.penalty", "1.0"));
    private static final String BACKEND = System.getProperty("qwen.harness.backend", "auto");
    // The dense FP16 model.onnx needs a large host heap; allow pointing at the smaller INT4 export for local runs.
    private static final String MODEL_FILE =
            System.getProperty("qwen.harness.modelfile", QwenModelDirValidator.DEFAULT_MODEL_FILE);

    private static final List<String> PROMPTS = List.of(
            "<|im_start|>user\nWrite a one-line Python function that adds two numbers.<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n");

    private static QwenInferenceEngine engine;
    private static Qwen2Runtime runtime;
    private static Qwen2Config config;
    private static QwenTokenizer tokenizer;

    static boolean harnessEnabled() {
        if (!"true".equalsIgnoreCase(System.getProperty("qwen.enable.experimental.runtime"))) {
            return false;
        }
        if (!QwenDecoderOnlyForwardPass.experimentalEnabled()) {
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
        assertNotNull(dir, "Model directory must resolve when the harness is enabled");
        engine = new QwenInferenceEngine(dir, MAX_TOKENS, BACKEND, MODEL_FILE);
        engine.initialize();
        assertTrue(engine.isReady(), "Engine must be ready");
        runtime = engine.getRuntime();
        config = engine.getConfig();
        tokenizer = QwenTokenizer.load(dir.resolve("tokenizer.json"));
        log.info("[harness] backend={}, modelFile={}, maxTokens={}, penalty={}",
                BACKEND, MODEL_FILE, MAX_TOKENS, PENALTY);
    }

    @AfterAll
    static void tearDown() {
        if (engine != null) {
            engine.shutdown();
        }
    }

    @Test
    void experimentalSessionMatchesProductionTokenForToken() {
        for (String prompt : PROMPTS) {
            compareOnePrompt(prompt);
        }
    }

    private void compareOnePrompt(String prompt) {
        // ── Production path (gold standard): collect streamed (non-stop) token ids + wall time ──
        runtime.setRepetitionPenalty(PENALTY);
        List<Integer> productionTokens = new ArrayList<>();
        long prodStart = System.nanoTime();
        runtime.generateStreaming(prompt, MAX_TOKENS, (tokenId, fullText, delta) -> productionTokens.add(tokenId));
        long prodNanos = System.nanoTime() - prodStart;
        // Production breaks before recording the stop token, so < MAX_TOKENS streamed means it hit a stop token.
        String productionFinish = productionTokens.size() < MAX_TOKENS ? "eos_token" : "length";

        // ── Experimental session path: same input ids, same maxTokens, same greedy+penalty ──
        int[] inputIds = tokenizer.encode(prompt);
        List<Integer> inputTokenIds = new ArrayList<>(inputIds.length);
        for (int id : inputIds) {
            inputTokenIds.add(id);
        }
        DecoderOnlyForwardPass forwardPass = new QwenDecoderOnlyForwardPass(config, runtime);
        DecoderOnlyGenerationLoop loop = new DecoderOnlyGenerationLoop(forwardPass, 0, 0);
        List<Integer> experimentalStreamed = new ArrayList<>();
        long expStart = System.nanoTime();
        DecoderOnlyGenerationResult result = loop.generate(
                inputTokenIds, MAX_TOKENS, QwenTokenSelector.greedy(PENALTY), experimentalStreamed::add);
        long expNanos = System.nanoTime() - expStart;

        log.info("[harness] prompt='{}'", prompt.replace("\n", "\\n"));
        log.info("[harness] production : {} tokens, finish={}, {} ms, ids={}",
                productionTokens.size(), productionFinish, prodNanos / 1_000_000L, productionTokens);
        log.info("[harness] experimental: {} streamed (gen={}), finish={}, {} ms, gen-ids={}",
                experimentalStreamed.size(), result.tokensGenerated(), result.finishReason(),
                expNanos / 1_000_000L, result.generatedTokenIds());

        // ── Hard gates (harmonised contract — no stop-token special case): generated ids, streamed ids and
        //    finish reason must match the production runtime exactly ──
        assertEquals(productionTokens, result.generatedTokenIds(),
                "Generated token ids must match the production runtime token-for-token");
        assertEquals(productionTokens, experimentalStreamed,
                "Streamed tokens must match the production runtime token-for-token");
        assertEquals(productionFinish, result.finishReason(), "Finish reason must match");

        // The terminating stop token is reported separately, not as a generated token.
        if ("eos_token".equals(result.finishReason())) {
            assertTrue(QwenStopTokenPolicy.shouldStop(result.finishTokenId()),
                    "finishTokenId must be the Qwen stop token that terminated generation");
        }
    }
}
