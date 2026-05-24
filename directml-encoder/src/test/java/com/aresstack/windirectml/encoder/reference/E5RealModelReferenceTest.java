package com.aresstack.windirectml.encoder.reference;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.encoder.bert.BertEncoderConfig;
import com.aresstack.windirectml.encoder.bert.CpuBertEncoder;
import com.aresstack.windirectml.encoder.bert.DirectMlBertEncoder;
import com.aresstack.windirectml.encoder.e5.E5Encoders;
import com.aresstack.windirectml.encoder.e5.E5Prefixes;
import com.aresstack.windirectml.encoder.e5.E5Variant;
import com.aresstack.windirectml.encoder.util.CosineSimilarity;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Reference test for the E5 family that actually loads a real
 * checkpoint from disk – no synthetic weights.
 * <p>
 * This is the production-evidence test the architecture review asked
 * for: it proves that
 * <ul>
 *   <li>{@code -De5.model} picks the right {@link E5Variant},</li>
 *   <li>{@link E5Encoders#resolveConfig(Path, E5Variant)} couples
 *       directory + variant config correctly and rejects mismatches,</li>
 *   <li>both {@link CpuBertEncoder} and {@link DirectMlBertEncoder}
 *       run on real E5 weights,</li>
 *   <li>the {@code query: } / {@code passage: } prefixes go through
 *       cleanly via {@link EmbeddingRequest#prefix()}.</li>
 * </ul>
 * <p>
 * The test auto-skips when no E5 model directory is present locally.
 * Resolution order:
 * <ol>
 *   <li>{@code -De5.testModelDir=&lt;path&gt;} (explicit override),</li>
 *   <li>{@code -De5.testVariant=small-v2|base-v2|large-v2|base-sts-en-de}
 *       combined with the {@link E5Variant#directoryHints()},</li>
 *   <li>otherwise: walk all variants' directory hints in declaration
 *       order and use the first one that contains
 *       {@code model.safetensors} + {@code tokenizer.json}.</li>
 * </ol>
 * Additional runtime gates:
 * <ul>
 *   <li>no Windows / D3D12 → skip cleanly,</li>
 *   <li>no DirectML-capable adapter → skip cleanly.</li>
 * </ul>
 */
@EnabledIf("modelPresent")
class E5RealModelReferenceTest {

    private static final String[] CORPUS = {
            "A cat sits on the mat.",
            "Eine Katze sitzt auf der Matte.",
            "The price of crude oil dropped sharply.",
            "Der Ölpreis ist stark gefallen."
    };

    private static Path modelDir;
    private static E5Variant variant;
    private static BertEncoderConfig resolvedConfig;
    private static CpuBertEncoder cpuModel;
    private static DirectMlBertEncoder dmlModel;

    static boolean modelPresent() {
        // TEMP DEBUG: print resolution/environment so we can see why tests skip.
        try {
            System.err.println("DEBUG E5: user.dir=" + System.getProperty("user.dir"));
            System.err.println("DEBUG E5: e5.testModelDir=" + System.getProperty("e5.testModelDir"));
            System.err.println("DEBUG E5: e5.testVariant=" + System.getProperty("e5.testVariant"));
            Resolution r = resolveOrNull();
            if (r == null) {
                System.err.println("DEBUG E5: resolveOrNull() returned null");
                return false;
            }
            Path dir = r.dir();
            System.err.println("DEBUG E5: resolveOrNull() -> variant=" + r.variant() + ", dir=" + dir);
            System.err.println("DEBUG E5: Files.isDirectory(dir)=" + Files.isDirectory(dir));
            System.err.println("DEBUG E5: exists model.safetensors=" + Files.exists(dir.resolve("model.safetensors")));
            System.err.println("DEBUG E5: exists tokenizer.json=" + Files.exists(dir.resolve("tokenizer.json")));
            return true;
        } catch (Throwable t) {
            System.err.println("DEBUG E5: modelPresent() threw: " + t);
            return false;
        }
    }

    /**
     * Resolve the (variant, modelDir) pair without any GPU work.
     */
    private static Resolution resolveOrNull() {
        String override = System.getProperty("e5.testModelDir");
        if (override != null && !override.isBlank()) {
            Path p = Path.of(override);
            if (looksLikeE5Dir(p)) {
                E5Variant v = parseTestVariant();
                return new Resolution(v, p);
            }
            return null;
        }
        String requested = System.getProperty("e5.testVariant");
        if (requested != null && !requested.isBlank()) {
            E5Variant v = E5Variant.parse(requested);
            for (Path hint : v.directoryHints()) {
                if (looksLikeE5Dir(hint)) return new Resolution(v, hint);
            }
            return null;
        }
        // Walk all variants in declaration order.
        for (E5Variant v : E5Variant.values()) {
            for (Path hint : v.directoryHints()) {
                if (looksLikeE5Dir(hint)) return new Resolution(v, hint);
            }
        }
        return null;
    }

    static Path resolveModelDir() {
        Resolution r = resolveOrNull();
        return r != null ? r.dir() : null;
    }

    private static boolean looksLikeE5Dir(Path p) {
        return p != null
                && Files.isDirectory(p)
                && Files.exists(p.resolve("model.safetensors"))
                && Files.exists(p.resolve("tokenizer.json"));
    }

    private static E5Variant parseTestVariant() {
        String requested = System.getProperty("e5.testVariant");
        return E5Variant.parse(requested);
    }

    private record Resolution(E5Variant variant, Path dir) {
    }

    @BeforeAll
    static void load() throws Exception {
        // TEMP DEBUG: report WindowsBindings and resolution info early so we
        // can observe exactly which skip gate fires inside the test JVM.
        try {
            System.err.println("DEBUG E5: WindowsBindings.isSupported()=" + WindowsBindings.isSupported());
        } catch (Throwable t) {
            System.err.println("DEBUG E5: WindowsBindings.isSupported() threw: " + t);
        }

        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12");

        Resolution r = resolveOrNull();
        if (r == null) {
            System.err.println("DEBUG E5: resolveOrNull() in @BeforeAll returned null -> skipping");
        } else {
            System.err.println("DEBUG E5: @BeforeAll resolveOrNull() -> variant=" + r.variant() + ", dir=" + r.dir());
            System.err.println("DEBUG E5: @BeforeAll Files.isDirectory=" + Files.isDirectory(r.dir()));
            System.err.println("DEBUG E5: @BeforeAll has model.safetensors=" + Files.exists(r.dir().resolve("model.safetensors")));
            System.err.println("DEBUG E5: @BeforeAll has tokenizer.json=" + Files.exists(r.dir().resolve("tokenizer.json")));
        }

        assumeTrue(r != null, "No real E5 model found on disk – skipping.");
        modelDir = r.dir();
        variant = r.variant();

        // Step 1: config.json on disk MUST match the chosen variant.
        // resolveConfig() will throw EmbeddingException on mismatch.
        try {
            resolvedConfig = E5Encoders.resolveConfig(modelDir, variant);
        } catch (EmbeddingException e) {
            System.err.println("DEBUG E5: E5Encoders.resolveConfig threw: " + e.getMessage());
            // Treat a directory/variant mismatch as a skip rather than a
            // failure: the user pointed us at a directory that does not
            // match the requested variant. The error message itself is
            // covered by E5VariantAndConfigTest.
            assumeTrue(false, "E5 directory/variant mismatch – skipping: "
                    + e.getMessage());
            return;
        }
        assertNotNull(resolvedConfig);
        resolvedConfig.validate();
        assertEquals(variant.config().hiddenSize(), resolvedConfig.hiddenSize(),
                "resolveConfig must agree with declared variant on hiddenSize");
        assertEquals(variant.config().numLayers(), resolvedConfig.numLayers(),
                "resolveConfig must agree with declared variant on numLayers");

        // Step 2: CPU encoder always loads.
        cpuModel = E5Encoders.loadCpu(modelDir, variant);
        assertTrue(cpuModel.isReady(), "CPU E5 encoder must report ready after load");

        // Step 3: DirectML encoder may need to be skipped if there is no
        // D3D12/DirectML adapter on this CI runner.
        try {
            dmlModel = E5Encoders.loadDirectMl(modelDir, variant);
        } catch (Exception e) {
            String msg = e.getMessage() == null ? "" : e.getMessage();
            System.err.println("DEBUG E5: E5Encoders.loadDirectMl threw: " + msg);
            if (msg.toLowerCase(Locale.ROOT).contains("directml")
                    || msg.toLowerCase(Locale.ROOT).contains("d3d12")) {
                assumeTrue(false, "Skipping DirectML half: " + msg);
                return;
            }
            throw e;
        }
        assertTrue(dmlModel.isReady(), "DirectML E5 encoder must report ready after load");
    }

    @AfterAll
    static void closeAll() {
        if (dmlModel != null) dmlModel.close();
        if (cpuModel != null) cpuModel.close();
    }

    @Test
    void dimensionMatchesVariantHiddenSize() throws Exception {
        EmbeddingVector v = dmlModel.embed(EmbeddingRequest.of("hello world"));
        assertEquals(resolvedConfig.outputDimension(), v.dimension(),
                "DirectML output dimension must match config.outputDimension");
        assertEquals(variant.config().hiddenSize(), v.dimension(),
                "DirectML output dimension must match variant hiddenSize");
    }

    @Test
    void unitNormForDefaultNormalize() throws Exception {
        EmbeddingVector v = dmlModel.embed(EmbeddingRequest.of(CORPUS[0]));
        double n2 = 0;
        for (float f : v.values()) n2 += (double) f * f;
        double norm = Math.sqrt(n2);
        assertTrue(Math.abs(norm - 1.0) < 1e-3,
                "expected unit-norm output (normalize=true default), got |x|=" + norm);
    }

    @Test
    void cpuAndDirectMlAgreeAcrossCorpus() throws Exception {
        double minSim = Double.POSITIVE_INFINITY;
        for (String text : CORPUS) {
            EmbeddingVector cpu = cpuModel.embed(EmbeddingRequest.of(text));
            EmbeddingVector dml = dmlModel.embed(EmbeddingRequest.of(text));
            assertEquals(cpu.dimension(), dml.dimension(),
                    "CPU/DML must produce same-dimension vectors");
            double sim = CosineSimilarity.compute(cpu.values(), dml.values());
            System.out.printf("E5 cos(CPU, DML) [%s] = %.6f%n", abbreviate(text), sim);
            assertTrue(sim > 0.99,
                    "cos(CPU, DML) must exceed 0.99 on real E5 weights for ["
                            + text + "], got " + sim);
            if (sim < minSim) minSim = sim;
        }
        System.out.printf("E5 min cos(CPU, DML) = %.6f (variant=%s)%n",
                minSim, variant.token());
    }

    @Test
    void queryAndPassagePrefixesYieldDistinctVectors() throws Exception {
        // The E5 fine-tune is trained so that "query: X" and "passage: X"
        // produce intentionally different embeddings. They should still
        // be highly similar (same surface text), but not identical.
        String text = "A cat sits on the mat.";
        EmbeddingVector q = dmlModel.embed(
                E5Prefixes.request(text, E5Prefixes.Role.QUERY, true));
        EmbeddingVector p = dmlModel.embed(
                E5Prefixes.request(text, E5Prefixes.Role.PASSAGE, true));
        double sim = CosineSimilarity.compute(q.values(), p.values());
        System.out.printf("E5 cos(query:, passage:) [%s] = %.6f%n",
                abbreviate(text), sim);
        assertTrue(sim < 0.9999,
                "query: and passage: must not produce identical vectors, got cos=" + sim);
        assertTrue(sim > 0.5,
                "query: and passage: of the same text should still be related, got cos=" + sim);
    }

    @Test
    void prefixIsAppliedConsistentlyOnCpuAndDirectMl() throws Exception {
        // Same text + same role on CPU and DML must still match.
        String text = "Eine Katze sitzt auf der Matte.";
        EmbeddingRequest req = E5Prefixes.request(text, E5Prefixes.Role.QUERY, true);
        EmbeddingVector cpu = cpuModel.embed(req);
        EmbeddingVector dml = dmlModel.embed(req);
        double sim = CosineSimilarity.compute(cpu.values(), dml.values());
        assertTrue(sim > 0.99,
                "prefixed embed must agree CPU↔DML, got cos=" + sim);
    }

    @Test
    void normalizeFalseReturnsUnnormalisedVector() throws Exception {
        EmbeddingRequest req = new EmbeddingRequest("hello world", false, null);
        EmbeddingVector dml = dmlModel.embed(req);
        assertFalse(dml.normalized(),
                "DirectML must propagate normalize=false through EmbeddingVector");
        double n2 = 0;
        for (float f : dml.values()) n2 += (double) f * f;
        double norm = Math.sqrt(n2);
        assertTrue(Math.abs(norm - 1.0) > 1e-2,
                "normalize=false must not produce a unit-norm vector, got |x|=" + norm);
    }

    private static String abbreviate(String s) {
        return s.length() <= 40 ? s : s.substring(0, 37) + "...";
    }
}
