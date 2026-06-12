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
 * Slice 9 — controlled Qwen production-vs-session benchmark (manual dev harness).
 *
 * <p>Measures the production {@link Qwen2Runtime} path against the experimental
 * {@link QwenDecoderOnlyForwardPass}/{@link DecoderOnlyGenerationLoop} session path on the SAME loaded model, same
 * prompt, same maxTokens and same greedy/penalty. Warmup runs precede measured runs, and measured runs alternate
 * production/session so warmup bias is visible per run. This measures only — it changes nothing in the production
 * path, and never switches a default.</p>
 *
 * <p><b>Never runs in normal CI.</b> Requires explicit opt-in and a local model. Manual run:</p>
 * <pre>{@code
 * ./gradlew :directml-inference:test --tests "*.qwen.QwenDecoderOnlySessionBenchmarkTest" \
 *     -Dqwen.decoderonly.benchmark=true \
 *     -Dqwen.decoderonly.experimental=true \
 *     -Dqwen.enable.experimental.runtime=true \
 *     -Dqwen.model.dir=<path-to>/qwen2.5-coder-0.5b-directml-int4 \
 *     -Dqwen.harness.modelfile=model_q4f16.onnx \
 *     -Dqwen.harness.maxtokens=64 --info
 * }</pre>
 */
@EnabledIf("benchmarkEnabled")
class QwenDecoderOnlySessionBenchmarkTest {

    private static final Logger log = LoggerFactory.getLogger(QwenDecoderOnlySessionBenchmarkTest.class);

    private static final int MAX_TOKENS = Integer.getInteger("qwen.harness.maxtokens", 64);
    private static final int WARMUP_RUNS = Integer.getInteger("qwen.harness.warmup", 1);
    private static final int MEASURE_RUNS = Integer.getInteger("qwen.harness.runs", 3);
    private static final float PENALTY = Float.parseFloat(System.getProperty("qwen.harness.penalty", "1.0"));
    private static final String BACKEND = System.getProperty("qwen.harness.backend", "auto");
    private static final String MODEL_FILE =
            System.getProperty("qwen.harness.modelfile", QwenModelDirValidator.DEFAULT_MODEL_FILE);

    private static final String PROMPT =
            "<|im_start|>user\nWrite a Python function that computes the factorial of n and briefly explain it."
                    + "<|im_end|>\n<|im_start|>assistant\n";

    private static QwenInferenceEngine engine;
    private static Qwen2Runtime runtime;
    private static Qwen2Config config;
    private static QwenTokenizer tokenizer;

    static boolean benchmarkEnabled() {
        if (!Boolean.getBoolean("qwen.decoderonly.benchmark")) {
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
        // Production runs go through generateStreaming, which now defaults to the session path (slice 11a); force the
        // legacy path so the production-vs-session comparison stays valid. runSession() drives the loop directly and
        // is unaffected.
        System.setProperty("qwen.runtime", "legacy");
        Path dir = resolveModelDir();
        assertNotNull(dir, "Model directory must resolve when the benchmark is enabled");
        engine = new QwenInferenceEngine(dir, MAX_TOKENS, BACKEND, MODEL_FILE);
        engine.initialize();
        assertTrue(engine.isReady(), "Engine must be ready");
        runtime = engine.getRuntime();
        config = engine.getConfig();
        tokenizer = QwenTokenizer.load(dir.resolve("tokenizer.json"));
    }

    @AfterAll
    static void tearDown() {
        if (engine != null) {
            engine.shutdown();
        }
        System.clearProperty("qwen.runtime");
    }

    @Test
    void benchmarkProductionVsSession() {
        int[] inputIds = tokenizer.encode(PROMPT);
        List<Integer> inputTokenIds = new ArrayList<>(inputIds.length);
        for (int id : inputIds) {
            inputTokenIds.add(id);
        }

        log.info("[bench] model={}, backend={}, promptTokens={}, maxTokens={}, penalty={}, warmup={}, runs={}",
                MODEL_FILE, BACKEND, inputIds.length, MAX_TOKENS, PENALTY, WARMUP_RUNS, MEASURE_RUNS);

        for (int w = 0; w < WARMUP_RUNS; w++) {
            runProduction();
            runSession(inputTokenIds);
        }

        List<Integer> productionTokens = null;
        List<Integer> sessionTokens = null;
        String productionFinish = null;
        String sessionFinish = null;
        long prodMsSum = 0;
        long sessMsSum = 0;
        int prodGen = 0;
        int sessGen = 0;

        for (int r = 0; r < MEASURE_RUNS; r++) {
            ProductionRun p = runProduction();
            SessionRun s = runSession(inputTokenIds);

            double prodTps = p.tokens.size() * 1000.0 / Math.max(1, p.totalMs);
            double sessTps = s.result.tokensGenerated() * 1000.0 / Math.max(1, s.totalMs);
            log.info("[bench] run {} production : gen={}, total={} ms, {} tok/s, finish={}",
                    r, p.tokens.size(), p.totalMs, String.format("%.1f", prodTps), p.finishReason);
            log.info("[bench] run {} session    : gen={}, total={} ms, {} tok/s, finish={}, prefill={} ms, decode={} ms",
                    r, s.result.tokensGenerated(), s.totalMs, String.format("%.1f", sessTps), s.result.finishReason(),
                    s.result.prefillNanos() / 1_000_000L, s.result.decoderStepNanos() / 1_000_000L);

            prodMsSum += p.totalMs;
            sessMsSum += s.totalMs;
            prodGen += p.tokens.size();
            sessGen += s.result.tokensGenerated();
            if (r == 0) {
                productionTokens = p.tokens;
                sessionTokens = s.result.generatedTokenIds();
                productionFinish = p.finishReason;
                sessionFinish = s.result.finishReason();
            }
        }

        // Production's own per-stage breakdown (prefill/decode) is logged by the runtime; surface it once here too.
        log.info("[bench] production last-profile:\n{}", runtime.getLastProfile());

        double prodMeanTps = prodGen * 1000.0 / Math.max(1, prodMsSum);
        double sessMeanTps = sessGen * 1000.0 / Math.max(1, sessMsSum);
        log.info("[bench] MEAN over {} runs: production {} ms/run, {} tok/s | session {} ms/run, {} tok/s",
                MEASURE_RUNS, prodMsSum / MEASURE_RUNS, String.format("%.1f", prodMeanTps),
                sessMsSum / MEASURE_RUNS, String.format("%.1f", sessMeanTps));

        // ── Correctness gates (independent of timing) ──
        assertEquals(productionTokens, sessionTokens,
                "Generated token ids must match the production runtime token-for-token");
        assertEquals(productionFinish, sessionFinish, "Finish reason must match the production runtime");
    }

    // ── path drivers ────────────────────────────────────────────────────────

    private record ProductionRun(List<Integer> tokens, long totalMs, String finishReason) {
    }

    private record SessionRun(DecoderOnlyGenerationResult result, long totalMs) {
    }

    private ProductionRun runProduction() {
        runtime.setRepetitionPenalty(PENALTY);
        List<Integer> tokens = new ArrayList<>();
        long start = System.nanoTime();
        runtime.generateStreaming(PROMPT, MAX_TOKENS, (tokenId, fullText, delta) -> tokens.add(tokenId));
        long totalMs = (System.nanoTime() - start) / 1_000_000L;
        // Production breaks before recording the stop token: fewer than MAX_TOKENS streamed means it hit a stop.
        String finishReason = tokens.size() < MAX_TOKENS ? "eos_token" : "length";
        return new ProductionRun(tokens, totalMs, finishReason);
    }

    private SessionRun runSession(List<Integer> inputTokenIds) {
        DecoderOnlyForwardPass forwardPass = new QwenDecoderOnlyForwardPass(config, runtime);
        DecoderOnlyGenerationLoop loop = new DecoderOnlyGenerationLoop(forwardPass, 0, 0);
        long start = System.nanoTime();
        DecoderOnlyGenerationResult result = loop.generate(
                inputTokenIds, MAX_TOKENS, QwenTokenSelector.greedy(PENALTY), null);
        long totalMs = (System.nanoTime() - start) / 1_000_000L;
        return new SessionRun(result, totalMs);
    }
}
