package com.aresstack.windirectml.encoder.reference;

import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.encoder.minilm.CpuMiniLmEncoder;
import com.aresstack.windirectml.encoder.minilm.DirectMlMiniLmEncoder;
import com.aresstack.windirectml.encoder.util.CosineSimilarity;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Vergleichstest CPU vs DirectML für den vollständigen
 * {@code all-MiniLM-L6-v2}-Encoder.
 * <p>
 * Wird automatisch deaktiviert, falls das Modell lokal nicht vorhanden
 * ist (gleicher Pfad wie {@link EmbeddingReferenceTest}). Zusätzliche
 * Skip-Bedingungen zur Laufzeit:
 * <ul>
 *   <li>kein D3D12-fähiges Windows,</li>
 *   <li>keine DirectML-fähige Karte,</li>
 *   <li>DirectML-DLL liefert kein {@code DML_FEATURE_LEVEL_5_1}.</li>
 * </ul>
 * Letzteres lässt sich per
 * {@code -Dwindirectml.directml.dll=<redistributable>} überbrücken.
 * <p>
 * Erwartung: pro Eingabesatz cosine(CPU, DirectML) &gt; 0.99 und
 * Ausgabedimension exakt 384. Die L2-Norm der DirectML-Ausgabe muss
 * 1 ± 1e-3 sein.
 */
@EnabledIf("modelPresent")
class DirectMlMiniLmEmbeddingReferenceTest {

    private static final String[] CORPUS = {
            "A cat sits on the mat.",
            "A feline rests on a rug.",
            "The price of oil dropped sharply.",
            "hello world"
    };

    private static CpuMiniLmEncoder cpuModel;
    private static DirectMlMiniLmEncoder dmlModel;

    static boolean modelPresent() {
        return resolveModelDir() != null;
    }

    static Path resolveModelDir() {
        String override = System.getProperty("minilm.testModelDir");
        if (override != null && !override.isBlank()) {
            Path p = Path.of(override);
            return Files.exists(p.resolve("model.safetensors")) ? p : null;
        }
        for (Path p : new Path[]{
                Path.of("model/all-MiniLM-L6-v2"),
                Path.of("../model/all-MiniLM-L6-v2"),
                Path.of("model/sentence-transformers/all-MiniLM-L6-v2")
        }) {
            if (Files.exists(p.resolve("model.safetensors"))
                    && Files.exists(p.resolve("tokenizer.json"))) {
                return p;
            }
        }
        return null;
    }

    @BeforeAll
    static void load() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12");
        Path dir = resolveModelDir();
        cpuModel = CpuMiniLmEncoder.load(dir);
        try {
            dmlModel = DirectMlMiniLmEncoder.load(dir);
        } catch (Exception e) {
            // Skip cleanly when the DLL/Adapter does not support FL 5.1.
            String msg = e.getMessage() == null ? "" : e.getMessage();
            if (msg.contains("DML_FEATURE_LEVEL_5_1")
                    || msg.contains("No DirectML device")
                    || msg.contains("DirectML requires Windows")) {
                assumeTrue(false, "Skipping: " + msg);
                return;
            }
            throw e;
        }
    }

    @AfterAll
    static void closeAll() {
        if (dmlModel != null) dmlModel.close();
        if (cpuModel != null) cpuModel.close();
    }

    @Test
    void dimensionAndUnitNorm() throws Exception {
        EmbeddingVector v = dmlModel.embed(EmbeddingRequest.of("hello world"));
        assertEquals(384, v.dimension());
        double n2 = 0;
        for (float f : v.values()) n2 += (double) f * f;
        double norm = Math.sqrt(n2);
        assertTrue(Math.abs(norm - 1.0) < 1e-3,
                "expected unit-norm output, got |x|=" + norm);
    }

    @Test
    void cosineSimilarityToCpuAboveThreshold() throws Exception {
        double minSim = Double.POSITIVE_INFINITY;
        for (String text : CORPUS) {
            EmbeddingVector cpu = cpuModel.embed(EmbeddingRequest.of(text));
            EmbeddingVector dml = dmlModel.embed(EmbeddingRequest.of(text));
            double sim = CosineSimilarity.compute(cpu.values(), dml.values());
            System.out.printf("cos(CPU, DML) [%s] = %.6f%n", abbreviate(text), sim);
            assertTrue(sim > 0.99,
                    "cosine(CPU, DML) must exceed 0.99 for [" + text + "], got " + sim);
            if (sim < minSim) minSim = sim;
        }
        System.out.printf("min cos(CPU, DML) across corpus = %.6f%n", minSim);
    }

    @Test
    void preservesSemanticOrderingLikeCpu() throws Exception {
        EmbeddingVector a = dmlModel.embed(EmbeddingRequest.of(CORPUS[0]));
        EmbeddingVector b = dmlModel.embed(EmbeddingRequest.of(CORPUS[1]));
        EmbeddingVector c = dmlModel.embed(EmbeddingRequest.of(CORPUS[2]));
        double similar = CosineSimilarity.compute(a.values(), b.values());
        double unrelated = CosineSimilarity.compute(a.values(), c.values());
        assertTrue(similar > unrelated + 0.2,
                "DirectML output must keep CPU ordering: similar=" + similar
                        + " vs unrelated=" + unrelated);
    }

    private static String abbreviate(String s) {
        return s.length() <= 32 ? s : s.substring(0, 29) + "...";
    }
}

