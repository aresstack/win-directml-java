package com.aresstack.windirectml.encoder.reference;

import com.aresstack.windirectml.encoder.reranker.BertCrossEncoderRerankers;
import com.aresstack.windirectml.encoder.reranker.CpuReranker;
import com.aresstack.windirectml.encoder.reranker.DirectMlReranker;
import com.aresstack.windirectml.encoder.reranker.RerankRequest;
import com.aresstack.windirectml.encoder.reranker.RerankResult;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Reference test for the BERT cross-encoder reranker family that
 * actually loads a real checkpoint from disk – no synthetic weights.
 * <p>
 * This is the production-evidence counterpart to
 * {@code E5RealModelReferenceTest} but for the reranker path: it
 * proves that
 * <ul>
 *   <li>{@link BertCrossEncoderRerankers#loadCpu(Path)} and
 *       {@link BertCrossEncoderRerankers#loadDirectMl(Path)} both
 *       accept real cross-encoder checkpoints,</li>
 *   <li>CPU and DirectML produce equivalent scores within a tight
 *       parity tolerance,</li>
 *   <li>the resulting ranking permutation is identical, so the
 *       sidecar's {@code rerank} method returns the same top-N order
 *       regardless of backend,</li>
 *   <li>the model assigns higher scores to clearly relevant documents
 *       than to clearly unrelated ones – a basic sanity check that
 *       the classifier head and pair tokenisation are wired correctly.</li>
 * </ul>
 * <p>
 * The test auto-skips when no reranker model directory is present
 * locally. Resolution order:
 * <ol>
 *   <li>{@code -Drerank.testModelDir=&lt;path&gt;} (explicit override),</li>
 *   <li>{@code model/cross-encoder-ms-marco-MiniLM-L-6-v2/},</li>
 *   <li>{@code model/cross-encoder/ms-marco-MiniLM-L-6-v2/},</li>
 *   <li>{@code model/ms-marco-MiniLM-L-6-v2/}.</li>
 * </ol>
 * Use {@code scripts/download-reranker.ps1} to fetch the default
 * {@code cross-encoder/ms-marco-MiniLM-L-6-v2} checkpoint.
 * <p>
 * Additional runtime gates:
 * <ul>
 *   <li>no Windows / D3D12 → skip cleanly,</li>
 *   <li>no DirectML-capable adapter → skip cleanly.</li>
 * </ul>
 */
@EnabledIf("modelPresent")
class RerankerRealModelReferenceTest {

    /** Score-parity tolerance between CPU and DirectML cross-encoder logits. */
    private static final double SCORE_TOLERANCE = 1.0e-2;

    private static final String QUERY = "What is DirectML?";

    /**
     * A small ranked corpus. The first two documents are clearly
     * relevant to the {@link #QUERY}; the rest are off-topic. The
     * concrete reranker scores are model-dependent, but the relative
     * ordering "relevant > unrelated" must hold for any reasonable
     * cross-encoder.
     */
    private static final String[] CORPUS = {
            "DirectML is a low-level Windows API for hardware-accelerated machine learning.",
            "DirectML lets ONNX Runtime execute neural networks on GPUs via Direct3D 12.",
            "Python is a high-level general-purpose programming language.",
            "The capital of France is Paris and it is famous for its cuisine."
    };

    private static Path modelDir;
    private static CpuReranker cpuModel;
    private static DirectMlReranker dmlModel;

    static boolean modelPresent() {
        return resolveModelDir() != null;
    }

    static Path resolveModelDir() {
        String override = System.getProperty("rerank.testModelDir");
        if (override != null && !override.isBlank()) {
            Path p = Path.of(override);
            return looksLikeRerankerDir(p) ? p : null;
        }
        for (Path candidate : new Path[]{
                Path.of("model/cross-encoder-ms-marco-MiniLM-L-6-v2"),
                Path.of("model/cross-encoder/ms-marco-MiniLM-L-6-v2"),
                Path.of("model/ms-marco-MiniLM-L-6-v2"),
                // Also probe one level up: the test working dir is the
                // module folder, so the repo-level model/ lives there.
                Path.of("../model/cross-encoder-ms-marco-MiniLM-L-6-v2"),
                Path.of("../model/cross-encoder/ms-marco-MiniLM-L-6-v2"),
        }) {
            if (looksLikeRerankerDir(candidate)) return candidate;
        }
        return null;
    }

    private static boolean looksLikeRerankerDir(Path p) {
        return p != null
                && Files.isDirectory(p)
                && Files.exists(p.resolve("model.safetensors"))
                && Files.exists(p.resolve("tokenizer.json"))
                && Files.exists(p.resolve("config.json"));
    }

    @BeforeAll
    static void load() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12");

        modelDir = resolveModelDir();
        assumeTrue(modelDir != null, "No real reranker model found on disk – skipping.");

        // Step 1: CPU loader always succeeds on a well-formed dir.
        cpuModel = BertCrossEncoderRerankers.loadCpu(modelDir);
        assertTrue(cpuModel.isReady(), "CPU reranker must report ready after load");
        assertNotNull(cpuModel.modelName(), "CPU reranker must expose modelName()");

        // Step 2: DirectML loader may need to be skipped if there is
        // no D3D12/DirectML adapter on this runner.
        try {
            dmlModel = BertCrossEncoderRerankers.loadDirectMl(modelDir);
        } catch (Exception e) {
            String msg = e.getMessage() == null ? "" : e.getMessage();
            if (msg.toLowerCase(Locale.ROOT).contains("directml")
                    || msg.toLowerCase(Locale.ROOT).contains("d3d12")) {
                assumeTrue(false, "Skipping DirectML half: " + msg);
                return;
            }
            throw e;
        }
        assertTrue(dmlModel.isReady(), "DirectML reranker must report ready after load");
    }

    @AfterAll
    static void closeAll() {
        if (dmlModel != null) dmlModel.close();
        if (cpuModel != null) cpuModel.close();
    }

    @Test
    void backendsAgreeOnModelName() {
        // Both loaders consume the same config.json -> same logical model.
        assertEquals(cpuModel.modelName(), dmlModel.modelName(),
                "CPU and DirectML reranker must report the same model name");
    }

    @Test
    void scoresAgreeBetweenCpuAndDirectMl() throws Exception {
        // Use topN = documents.length to get every document scored in
        // input order; that makes the per-index comparison trivial.
        RerankRequest req = new RerankRequest(QUERY, Arrays.asList(CORPUS), CORPUS.length);
        List<RerankResult> cpuRanked = cpuModel.rerank(req);
        List<RerankResult> dmlRanked = dmlModel.rerank(req);
        assertEquals(CORPUS.length, cpuRanked.size(),
                "CPU reranker must return one result per document");
        assertEquals(CORPUS.length, dmlRanked.size(),
                "DirectML reranker must return one result per document");

        // The results are sorted by descending score; align them by
        // originalIndex so we compare like with like.
        double[] cpuByIdx = new double[CORPUS.length];
        double[] dmlByIdx = new double[CORPUS.length];
        for (RerankResult r : cpuRanked) cpuByIdx[r.originalIndex()] = r.score();
        for (RerankResult r : dmlRanked) dmlByIdx[r.originalIndex()] = r.score();

        double maxAbsDiff = 0.0;
        for (int i = 0; i < CORPUS.length; i++) {
            double diff = Math.abs(cpuByIdx[i] - dmlByIdx[i]);
            System.out.printf(Locale.ROOT,
                    "rerank score [%d] cpu=%.6f dml=%.6f |diff|=%.6f%n",
                    i, cpuByIdx[i], dmlByIdx[i], diff);
            if (diff > maxAbsDiff) maxAbsDiff = diff;
        }
        System.out.printf(Locale.ROOT,
                "rerank max |cpu - dml| = %.6f (tolerance=%.6f)%n",
                maxAbsDiff, SCORE_TOLERANCE);
        assertTrue(maxAbsDiff < SCORE_TOLERANCE,
                "CPU and DirectML reranker scores must agree within "
                        + SCORE_TOLERANCE + ", got max |diff|=" + maxAbsDiff);
    }

    @Test
    void rankingOrderIsIdenticalAcrossBackends() throws Exception {
        RerankRequest req = new RerankRequest(QUERY, Arrays.asList(CORPUS), CORPUS.length);
        List<RerankResult> cpuRanked = cpuModel.rerank(req);
        List<RerankResult> dmlRanked = dmlModel.rerank(req);
        int[] cpuOrder = cpuRanked.stream().mapToInt(RerankResult::originalIndex).toArray();
        int[] dmlOrder = dmlRanked.stream().mapToInt(RerankResult::originalIndex).toArray();
        System.out.println("rerank CPU order = " + Arrays.toString(cpuOrder));
        System.out.println("rerank DML order = " + Arrays.toString(dmlOrder));
        assertArrayEqualsWithMsg(cpuOrder, dmlOrder,
                "CPU and DirectML reranker must produce identical top-N order");
    }

    @Test
    void relevantDocumentBeatsUnrelatedDocument() throws Exception {
        // Sanity check on the *direction* of the cross-encoder: a doc
        // about DirectML must outrank one about French cuisine on the
        // DirectML query, on both backends.
        RerankRequest req = new RerankRequest(QUERY, Arrays.asList(CORPUS), CORPUS.length);
        List<RerankResult> cpuRanked = cpuModel.rerank(req);
        List<RerankResult> dmlRanked = dmlModel.rerank(req);
        int cpuTop = cpuRanked.get(0).originalIndex();
        int dmlTop = dmlRanked.get(0).originalIndex();
        assertTrue(cpuTop == 0 || cpuTop == 1,
                "CPU reranker top-1 must be a DirectML document (idx 0/1), got " + cpuTop);
        assertTrue(dmlTop == 0 || dmlTop == 1,
                "DirectML reranker top-1 must be a DirectML document (idx 0/1), got " + dmlTop);
        // And the unrelated docs (idx 2, 3) must score below the
        // relevant docs (idx 0, 1) on at least one backend.
        double[] cpuByIdx = new double[CORPUS.length];
        for (RerankResult r : cpuRanked) cpuByIdx[r.originalIndex()] = r.score();
        double minRelevant = Math.min(cpuByIdx[0], cpuByIdx[1]);
        double maxUnrelated = Math.max(cpuByIdx[2], cpuByIdx[3]);
        assertTrue(minRelevant > maxUnrelated,
                "CPU reranker must score relevant docs above unrelated ones; "
                        + "minRelevant=" + minRelevant + " maxUnrelated=" + maxUnrelated);
    }

    @Test
    void topNTrimsRanking() throws Exception {
        // topN=2 must return exactly the two highest-scoring docs in
        // descending order – mirroring the sidecar's "results" array.
        RerankRequest req = new RerankRequest(QUERY, Arrays.asList(CORPUS), 2);
        List<RerankResult> ranked = dmlModel.rerank(req);
        assertEquals(2, ranked.size(), "topN=2 must trim to two entries");
        assertTrue(ranked.get(0).score() >= ranked.get(1).score(),
                "results must be sorted by descending score");
    }

    private static void assertArrayEqualsWithMsg(int[] expected, int[] actual, String msg) {
        if (!Arrays.equals(expected, actual)) {
            throw new AssertionError(msg
                    + " – expected " + Arrays.toString(expected)
                    + " but got " + Arrays.toString(actual));
        }
    }
}

