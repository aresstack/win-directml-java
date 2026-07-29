package com.aresstack.windirectml.runtime.facade;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.pack.EncoderPackageLifecycle;
import com.aresstack.windirectml.encoder.reranker.RerankResult;
import com.aresstack.windirectml.windows.WindowsBindings;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/**
 * Real-model investigation for the L12 cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L12-v2).
 *
 * <p>Two facts are asserted, both through the public {@link LocalMlRuntime} facade:
 * <ol>
 *   <li><b>Ranking works</b>: compile {@code reranker.wdmlpack}, load, and score
 *       {@code score(relevant) &gt; score(off-topic)} on CPU (and WARP when a device is present).</li>
 *   <li><b>Package-only is NOT yet satisfied</b>: loading from a directory that has the package +
 *       tokenizer/config but no {@code model.safetensors} fails, because the reranker load path
 *       ({@code BertCrossEncoderRerankers.REQUIRED_FILES}) still requires the raw weights to be
 *       present. Per the certification rule (package-only load is mandatory) L12 therefore stays
 *       UNVERIFIED in the catalog — see problems.md for the precise cause and fix.</li>
 * </ol>
 *
 * Opt-in: {@code -Dwindirectml.rerank.l12.dir=<downloaded-dir>} (forwarded by the root build).
 */
class RerankerL12CertificationIT {

    private static final String QUERY = "PF4J plugin framework";
    private static final String DOC_RELEVANT = "PF4J is a plugin framework for Java applications.";
    private static final String DOC_OFFTOPIC = "Tomatensuppe wird aus Tomaten gekocht.";
    private static final String[] RAW_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".onnx", ".onnx.data", ".pt"};

    private static Path rawDir() {
        String dirProp = System.getProperty("windirectml.rerank.l12.dir");
        assumeTrue(dirProp != null && !dirProp.isBlank(),
                "windirectml.rerank.l12.dir not set — opt-in L12 reranker investigation");
        Path dir = Path.of(dirProp);
        assertTrue(Files.isDirectory(dir), "rerank.l12.dir does not exist: " + dir);
        return dir;
    }

    @Test
    void l12RankingIsCorrectOnCpuAndWarp() throws Exception {
        Path rawDir = rawDir();
        EncoderPackageLifecycle.reranker().convert(rawDir, true);
        assertTrue(Files.isRegularFile(rawDir.resolve("reranker.wdmlpack")),
                "convert did not produce reranker.wdmlpack");

        RankOutcome cpu = rankOnce(Backend.CPU, rawDir);
        assertRankingSane(cpu, "CPU");

        if (WindowsBindings.isSupported()) {
            RankOutcome warp = rankOnce(Backend.WARP, rawDir);
            assertRankingSane(warp, "WARP");
            assertEquals(cpu.topOriginalIndex, warp.topOriginalIndex,
                    "CPU and WARP disagree on the top-ranked document");
        }
    }

    @Test
    void packageOnlyLoadIsNotYetSupported() throws Exception {
        // Documents the reason L12 stays UNVERIFIED: the reranker requires model.safetensors present
        // even though weights load from reranker.wdmlpack, so a package-only directory cannot load.
        Path rawDir = rawDir();
        EncoderPackageLifecycle.reranker().convert(rawDir, true);
        Path pkgDir = Files.createTempDirectory("wdml-reranker-l12-pkgonly-");
        copyPackageOnly(rawDir, pkgDir);
        assertTrue(Files.isRegularFile(pkgDir.resolve("reranker.wdmlpack")));
        assertNoRawWeights(pkgDir);

        EmbeddingException ex = assertThrows(EmbeddingException.class, () -> rankOnce(Backend.CPU, pkgDir));
        assertTrue(ex.getMessage() != null && ex.getMessage().toLowerCase(Locale.ROOT).contains("incomplete"),
                "expected an incomplete-model-dir error, got: " + ex.getMessage());
    }

    private static void assertRankingSane(RankOutcome outcome, String backend) {
        assertEquals(0, outcome.topOriginalIndex,
                backend + ": expected the relevant PF4J document to rank first");
        assertTrue(outcome.relevantScore > outcome.offtopicScore,
                backend + ": score(relevant)=" + outcome.relevantScore
                        + " must exceed score(off-topic)=" + outcome.offtopicScore);
    }

    private static RankOutcome rankOnce(Backend backend, Path dir) throws Exception {
        LocalMlRuntimeConfig config = LocalMlRuntimeConfig.builder().backend(backend).build();
        LocalMlRuntime runtime = LocalMlRuntime.create(config);
        try (LocalRerankerModel reranker = runtime.loadRerankerModel(new RerankerModelConfig(dir))) {
            List<RerankResult> results = reranker.rerank(QUERY, List.of(DOC_RELEVANT, DOC_OFFTOPIC));
            assertNotNull(results);
            assertEquals(2, results.size(), "expected two reranked results");
            return new RankOutcome(results.get(0).originalIndex(), scoreFor(results, 0), scoreFor(results, 1));
        }
    }

    private static double scoreFor(List<RerankResult> results, int originalIndex) {
        for (RerankResult r : results) {
            if (r.originalIndex() == originalIndex) {
                return r.score();
            }
        }
        throw new AssertionError("no result for original index " + originalIndex);
    }

    private static void copyPackageOnly(Path rawDir, Path pkgDir) throws IOException {
        try (Stream<Path> entries = Files.list(rawDir)) {
            List<Path> files = new ArrayList<>();
            entries.filter(Files::isRegularFile).forEach(files::add);
            for (Path file : files) {
                if (!isRawWeight(file)) {
                    Files.copy(file, pkgDir.resolve(file.getFileName().toString()));
                }
            }
        }
    }

    private static void assertNoRawWeights(Path dir) throws IOException {
        try (Stream<Path> entries = Files.list(dir)) {
            List<String> offenders = new ArrayList<>();
            entries.filter(Files::isRegularFile).filter(RerankerL12CertificationIT::isRawWeight)
                    .forEach(p -> offenders.add(p.getFileName().toString()));
            assertTrue(offenders.isEmpty(), "package-only dir must contain no raw weights: " + offenders);
        }
    }

    private static boolean isRawWeight(Path file) {
        String name = file.getFileName().toString().toLowerCase(Locale.ROOT);
        for (String suffix : RAW_WEIGHT_SUFFIXES) {
            if (name.endsWith(suffix)) {
                return true;
            }
        }
        return false;
    }

    private static final class RankOutcome {
        final int topOriginalIndex;
        final double relevantScore;
        final double offtopicScore;

        RankOutcome(int topOriginalIndex, double relevantScore, double offtopicScore) {
            this.topOriginalIndex = topOriginalIndex;
            this.relevantScore = relevantScore;
            this.offtopicScore = offtopicScore;
        }
    }
}
