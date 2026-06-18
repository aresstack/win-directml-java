package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.InferenceRequest;
import com.aresstack.windirectml.inference.InferenceResult;
import com.aresstack.windirectml.inference.prompt.PromptTask;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * T5-CORRECTNESS-CERT-1: certify the T5 WARP/AUTO <em>mixed</em> runtime (dense projections on DirectML, norms /
 * attention-softmax / relative-position-bias on the CPU reference path) against the validated CPU reference, taking
 * the reference as ground truth.
 *
 * <p>Two angles, both opt-in so the standard test run stays light:</p>
 * <ul>
 *   <li><b>Synthetic device cert</b> ({@code -Dt5.correctness.cert=true}, requires a D3D12/DirectML device): builds a
 *       tiny T5 {@code .wdmlpack} and runs the <em>same</em> package through {@link T5Runtime#load} (reference) and
 *       {@link T5Runtime#loadWarp} (WARP mixed boundary), then compares the greedy output token ids and the LM-head
 *       logits. This isolates the WARP dense-projection arithmetic from any tokenizer/model-download dependency, so it
 *       runs on this host today.</li>
 *   <li><b>Real-model cert</b> ({@code -Dt5.realModel=true} and a local model dir): drives the four curated T5 families
 *       through the {@link T5InferenceEngine} on {@code reference} vs {@code warp} and compares token ids + text. It
 *       {@linkplain #anyRealModelPresent skips cleanly} when no artifact is present (the default on CI and on this
 *       host — no T5 package is checked in).</li>
 * </ul>
 */
class T5MixedRuntimeCorrectnessCertTest {

    @TempDir
    Path tempDir;

    // ---- Synthetic device cert: reference vs WARP mixed path on the same package -------------------------------

    @Test
    @EnabledIf("syntheticCertEnabled")
    void warpMixedRuntimeMatchesReferenceTokenIdsOnSyntheticPackage() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML for the WARP mixed path");

        for (boolean gated : new boolean[]{false, true}) {
            T5Config config = gated ? T5TestFixtures.tinyConfig(true) : T5TestFixtures.tinyConfig(false);
            T5RuntimePackage pkg = compileTinyPackage(config, "synthetic-" + gated);

            int[] inputTokens = {1, 2, 3};
            int maxNewTokens = 6;

            T5Runtime reference = T5Runtime.load(pkg);
            int[] referenceTokens;
            try {
                referenceTokens = reference.generate(T5RuntimeRequest.greedy(
                        inputTokens, maxNewTokens, config.specialTokens())).outputTokenIds();
            } finally {
                reference.close();
            }

            int[] warpTokens;
            try (WindowsBindings bindings = new WindowsBindings()) {
                bindings.init("warp");
                assumeTrue(bindings.hasDirectMl(), "WARP backend did not provide a DirectML device");
                T5Runtime warp = T5Runtime.loadWarp(pkg, bindings);
                try {
                    assertTrue(warp.executionMode().contains("warp"),
                            "loadWarp must report a WARP execution mode, was: " + warp.executionMode());
                    warpTokens = warp.generate(T5RuntimeRequest.greedy(
                            inputTokens, maxNewTokens, config.specialTokens())).outputTokenIds();
                } finally {
                    warp.close();
                }
            }

            int firstDivergent = firstDivergentIndex(referenceTokens, warpTokens);
            System.out.println("[T5-CERT synthetic gated=" + gated + "] reference=" + Arrays.toString(referenceTokens)
                    + " warp=" + Arrays.toString(warpTokens) + " firstDivergent="
                    + (firstDivergent < 0 ? "none" : firstDivergent));

            if (firstDivergent >= 0) {
                fail("WARP mixed path diverged from the CPU reference (gated=" + gated + ") at greedy step "
                        + firstDivergent + ": reference=" + Arrays.toString(referenceTokens)
                        + " warp=" + Arrays.toString(warpTokens));
            }
        }
    }

    @Test
    @EnabledIf("syntheticCertEnabled")
    void warpLmHeadLogitsMatchReferenceWithinTolerance() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML for the WARP LM head");

        T5Config config = T5TestFixtures.tinyConfig(false);
        T5RuntimePackage pkg = compileTinyPackage(config, "lm-head");
        T5Weights weights = pkg.weights();

        int hidden = weights.lmHead().dims()[1] == 0 ? 0 : (int) weights.lmHead().dims()[1];
        Random random = new Random(42L);
        float[] hiddenState = new float[hidden];
        for (int i = 0; i < hidden; i++) {
            hiddenState[i] = (random.nextFloat() - 0.5f) * 2.0f;
        }

        T5LogitProjector reference = T5LmHead.from(weights);
        float[] referenceLogits = reference.logits(hiddenState);

        try (WindowsBindings bindings = new WindowsBindings()) {
            bindings.init("warp");
            assumeTrue(bindings.hasDirectMl(), "WARP backend did not provide a DirectML device");
            try (T5WarpLmHead warp = T5WarpLmHead.from(bindings, weights)) {
                float[] warpLogits = warp.logits(hiddenState);

                double maxAbsDiff = 0.0;
                int referenceTop1 = argmax(referenceLogits);
                int warpTop1 = argmax(warpLogits);
                for (int i = 0; i < referenceLogits.length; i++) {
                    maxAbsDiff = Math.max(maxAbsDiff, Math.abs(referenceLogits[i] - warpLogits[i]));
                }
                System.out.println("[T5-CERT lm-head] maxAbsLogitDiff=" + maxAbsDiff
                        + " referenceTop1=" + referenceTop1 + " warpTop1=" + warpTop1);

                assertTrue(referenceTop1 == warpTop1,
                        "WARP LM-head top-1 must match reference: reference=" + referenceTop1 + " warp=" + warpTop1);
                assertTrue(maxAbsDiff < 1e-2,
                        "WARP LM-head logits diverged from reference: maxAbsDiff=" + maxAbsDiff);
            }
        }
    }

    // ---- Real-model cert: the four curated T5 families, reference vs WARP --------------------------------------

    @Test
    @EnabledIf("anyRealModelPresent")
    void realT5ModelsMatchReferenceOnWarp() {
        List<String> report = new ArrayList<>();
        List<String> divergences = new ArrayList<>();
        boolean warpUsable = WindowsBindings.isSupported();
        for (RealModel model : REAL_MODELS) {
            Path dir = model.resolveDir();
            if (dir == null) {
                report.add(model.modelId + ": SKIPPED (no local artifact)");
                continue;
            }
            EngineRun reference;
            try {
                reference = runEngine(dir, model, "reference");
            } catch (Exception e) {
                report.add(model.modelId + ": reference FAILED (" + e.getMessage() + ")");
                divergences.add(model.modelId + ": reference engine failed: " + e.getMessage());
                continue;
            }
            if (!warpUsable) {
                report.add(model.modelId + ": reference ok (mode=" + reference.executionMode
                        + ", tokens=" + reference.tokenPreview + "); WARP SKIPPED (no D3D12/DirectML device)");
                continue;
            }
            try {
                EngineRun warp = runEngine(dir, model, "warp");
                boolean tokensMatch = reference.tokenPreview.equals(warp.tokenPreview);
                boolean textMatch = reference.text.equals(warp.text);
                report.add(model.modelId + ": mode=" + warp.executionMode
                        + " tokensMatch=" + tokensMatch + " textMatch=" + textMatch
                        + "\n      referenceTokens=" + reference.tokenPreview
                        + "\n      warpTokens     =" + warp.tokenPreview);
                // Greedy (temperature=0) -> the WARP mixed path must reproduce the CPU reference exactly.
                if (!tokensMatch || !textMatch) {
                    divergences.add(model.modelId + ": tokensMatch=" + tokensMatch + " textMatch=" + textMatch
                            + " (reference=" + reference.tokenPreview + " / warp=" + warp.tokenPreview + ")");
                }
            } catch (Exception e) {
                report.add(model.modelId + ": WARP FAILED (" + e.getMessage() + ")");
                divergences.add(model.modelId + ": WARP engine failed: " + e.getMessage());
            }
        }
        System.out.println("[T5-CERT real-model]\n  " + String.join("\n  ", report));
        assertTrue(report.stream().anyMatch(line -> !line.contains("SKIPPED")),
                "Expected at least one real T5 model present when the real-model cert is enabled:\n  "
                        + String.join("\n  ", report));
        // Lock the certification: any present model that diverges from the CPU reference fails the cert.
        if (!divergences.isEmpty()) {
            fail("T5 WARP mixed path diverged from the CPU reference for:\n  " + String.join("\n  ", divergences));
        }
    }

    private static EngineRun runEngine(Path dir, RealModel model, String backend) throws Exception {
        T5InferenceEngine engine = new T5InferenceEngine(dir, 20, backend);
        try {
            engine.initialize();
            InferenceResult result = engine.generate(InferenceRequest.builder()
                    .modelId(model.modelId)
                    .task(model.task)
                    .userPrompt(model.prompt)
                    .maxTokens(20)
                    .temperature(0.0f)
                    .build());
            return new EngineRun(engine.executionMode(), engine.lastOutputTokenPreview(),
                    result.getText() == null ? "" : result.getText());
        } finally {
            engine.shutdown();
        }
    }

    private static final class EngineRun {
        final String executionMode;
        final String tokenPreview;
        final String text;

        EngineRun(String executionMode, String tokenPreview, String text) {
            this.executionMode = executionMode;
            this.tokenPreview = tokenPreview;
            this.text = text;
        }
    }

    // ---- Fixtures / helpers -----------------------------------------------------------------------------------

    private T5RuntimePackage compileTinyPackage(T5Config config, String name) throws Exception {
        Path modelDir = tempDir.resolve("model-" + name + "-" + System.nanoTime());
        T5TestFixtures.writeConfig(modelDir, config);
        T5TestFixtures.writeSafeTensors(modelDir, T5TestFixtures.completeDenseT5Tensors(config));
        Path output = tempDir.resolve("cert-" + name + ".wdmlpack");
        T5WdmlPackCompiler.compile(new T5CompileOptions(modelDir, output, false, false));
        return T5RuntimePackage.open(output);
    }

    private static int firstDivergentIndex(int[] reference, int[] candidate) {
        int min = Math.min(reference.length, candidate.length);
        for (int i = 0; i < min; i++) {
            if (reference[i] != candidate[i]) {
                return i;
            }
        }
        return reference.length == candidate.length ? -1 : min;
    }

    private static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }

    // ---- Gating -----------------------------------------------------------------------------------------------

    @SuppressWarnings("unused")
    static boolean syntheticCertEnabled() {
        return Boolean.getBoolean("t5.correctness.cert");
    }

    @SuppressWarnings("unused")
    static boolean anyRealModelPresent() {
        if (!Boolean.getBoolean("t5.realModel")) {
            return false;
        }
        for (RealModel model : REAL_MODELS) {
            if (model.resolveDir() != null) {
                return true;
            }
        }
        return false;
    }

    /** The primary cert model (T5-REALMODEL-CERT-PREP-1): smallest, plain T5, fewest tokenizer special cases. */
    static final String PRIMARY_MODEL_ID = "google-t5/t5-small";

    private static final RealModel[] REAL_MODELS = {
            new RealModel(PRIMARY_MODEL_ID, PromptTask.SUMMARIZE,
                    "translate English to German: The house is wonderful.",
                    "t5-small", "google-t5/t5-small"),
            new RealModel("google/flan-t5-small", PromptTask.NONE,
                    "Answer the question: what is the capital of France?",
                    "flan-t5-small", "google/flan-t5-small"),
            new RealModel("Salesforce/codet5-small", PromptTask.SUMMARIZE,
                    "def add(a, b):\n    return a + b",
                    "codet5-small", "Salesforce/codet5-small"),
            new RealModel("Salesforce/codet5-base-multi-sum", PromptTask.SUMMARIZE,
                    "def add(a, b):\n    return a + b",
                    "codet5-base-multi-sum", "Salesforce/codet5-base-multi-sum"),
    };

    private static final class RealModel {
        final String modelId;
        final PromptTask task;
        final String prompt;
        final String[] relativeDirs;

        RealModel(String modelId, PromptTask task, String prompt, String... relativeDirs) {
            this.modelId = modelId;
            this.task = task;
            this.prompt = prompt;
            this.relativeDirs = relativeDirs;
        }

        Path resolveDir() {
            // 1. Per-model override: -Dt5.testModelDir.<modelId> (verbatim id, e.g. t5.testModelDir.google-t5/t5-small).
            Path perModel = validDir(System.getProperty("t5.testModelDir." + modelId));
            if (perModel != null) {
                return perModel;
            }
            // 2. Generic override -Dt5.testModelDir applies to the primary cert model only (so a single dir cannot be
            //    mis-applied to a different model id). Mirrors T5RealModelReferenceTest's convention.
            if (PRIMARY_MODEL_ID.equals(modelId)) {
                Path generic = validDir(System.getProperty("t5.testModelDir"));
                if (generic != null) {
                    return generic;
                }
            }
            // 3. Auto-resolution at the standard download location (<modelRoot>/<localDir>, default <repo>/model/...).
            for (String relative : relativeDirs) {
                for (Path base : new Path[]{Path.of("model", relative), Path.of("../model", relative)}) {
                    if (Files.isRegularFile(base.resolve("config.json"))) {
                        return base;
                    }
                }
            }
            return null;
        }

        private static Path validDir(String value) {
            if (value == null || value.isBlank()) {
                return null;
            }
            Path dir = Path.of(value);
            return Files.isRegularFile(dir.resolve("config.json")) ? dir : null;
        }
    }
}
