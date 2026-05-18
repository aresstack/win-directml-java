package com.aresstack.windirectml.encoder.minilm;

import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.encoder.EncoderTokenizer;
import com.aresstack.windirectml.encoder.util.CosineSimilarity;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Synthetischer Forward-Pass-Test für {@link CpuMiniLmEncoder}.
 * <p>
 * Verwendet eine winzige Architektur mit zufälligen Gewichten:
 * {@code hidden=12, layers=1, heads=2, headDim=6, intermediate=24, vocab=10}.
 * <p>
 * Verifiziert nur strukturelle Eigenschaften, da die Gewichte zufällig sind:
 * <ul>
 *   <li>Ausgabedimension korrekt</li>
 *   <li>Werte sind endlich (keine NaN/Inf)</li>
 *   <li>Identische Eingabe → identische Ausgabe (Determinismus)</li>
 *   <li>L2-Norm == 1 bei normalisierter Ausgabe</li>
 * </ul>
 */
class CpuMiniLmEncoderSyntheticTest {

    private static class TinyTokenizer implements EncoderTokenizer {
        @Override
        public Encoded encode(String text) {
            int n = Math.min(text.length() + 2, 6); // CLS + chars + SEP, capped
            int[] ids = new int[n];
            int[] mask = new int[n];
            int[] segs = new int[n];
            ids[0] = 2; // CLS
            for (int i = 1; i < n - 1; i++) ids[i] = 4 + (text.charAt(i - 1) % 4);
            ids[n - 1] = 3; // SEP
            for (int i = 0; i < n; i++) mask[i] = 1;
            return new Encoded(ids, mask, segs);
        }

        @Override
        public int padTokenId() {
            return 0;
        }

        @Override
        public int clsTokenId() {
            return 2;
        }

        @Override
        public int sepTokenId() {
            return 3;
        }

        @Override
        public int vocabSize() {
            return 10;
        }
    }

    private CpuMiniLmEncoder buildTinyEncoder(long seed) {
        MiniLmConfig cfg = new MiniLmConfig(
                /* hidden            */ 12,
                /* numLayers         */ 1,
                /* numHeads          */ 2,
                /* intermediate      */ 24,
                /* maxPos            */ 16,
                /* typeVocab         */ 2,
                /* vocab             */ 10,
                /* layerNormEps      */ 1e-12f,
                /* hiddenAct         */ "gelu",
                /* outputDimension   */ 12);
        MiniLmArchitecture arch = new MiniLmArchitecture(cfg);
        Random r = new Random(seed);

        float[] wordEmb = rand(r, cfg.vocabSize() * cfg.hiddenSize(), 0.02f);
        float[] posEmb = rand(r, cfg.maxPositionEmbeddings() * cfg.hiddenSize(), 0.02f);
        float[] ttEmb = rand(r, cfg.typeVocabSize() * cfg.hiddenSize(), 0.02f);
        float[] embLnG = ones(cfg.hiddenSize());
        float[] embLnB = zeros(cfg.hiddenSize());

        List<CpuMiniLmWeights.LayerWeights> layers = new ArrayList<>();
        int H = cfg.hiddenSize();
        int I = cfg.intermediateSize();
        for (int l = 0; l < cfg.numLayers(); l++) {
            layers.add(CpuMiniLmWeights.layerForTesting(
                    rand(r, H * H, 0.05f), zeros(H),
                    rand(r, H * H, 0.05f), zeros(H),
                    rand(r, H * H, 0.05f), zeros(H),
                    rand(r, H * H, 0.05f), zeros(H),
                    ones(H), zeros(H),
                    rand(r, I * H, 0.05f), zeros(I),
                    rand(r, H * I, 0.05f), zeros(H),
                    ones(H), zeros(H)));
        }

        CpuMiniLmWeights w = CpuMiniLmWeights.forTesting(arch, wordEmb, posEmb, ttEmb,
                embLnG, embLnB, layers);
        return new CpuMiniLmEncoder(arch, w, new TinyTokenizer());
    }

    @Test
    void embeddingHasExpectedShapeAndIsFinite() throws Exception {
        CpuMiniLmEncoder enc = buildTinyEncoder(42L);
        EmbeddingVector v = enc.embed(EmbeddingRequest.of("hello"));
        assertEquals(12, v.dimension());
        assertEquals(12, v.values().length);
        for (float f : v.values()) {
            assertTrue(Float.isFinite(f), "value not finite: " + f);
        }
    }

    @Test
    void normalizedVectorHasUnitNorm() throws Exception {
        CpuMiniLmEncoder enc = buildTinyEncoder(7L);
        EmbeddingVector v = enc.embed(new EmbeddingRequest("hello world", true, null));
        double norm = 0;
        for (float f : v.values()) norm += (double) f * f;
        assertEquals(1.0, Math.sqrt(norm), 1e-5);
    }

    @Test
    void sameInputProducesSameOutput() throws Exception {
        CpuMiniLmEncoder enc = buildTinyEncoder(123L);
        EmbeddingVector a = enc.embed(EmbeddingRequest.of("abc"));
        EmbeddingVector b = enc.embed(EmbeddingRequest.of("abc"));
        // Cosine == 1 für identische Vektoren
        assertEquals(1.0, CosineSimilarity.compute(a.values(), b.values()), 1e-6);
    }

    @Test
    void differentInputProducesDifferentOutput() throws Exception {
        CpuMiniLmEncoder enc = buildTinyEncoder(1L);
        EmbeddingVector a = enc.embed(EmbeddingRequest.of("aaaaa"));
        EmbeddingVector b = enc.embed(EmbeddingRequest.of("zzzzz"));
        double cos = CosineSimilarity.compute(a.values(), b.values());
        // Mit zufälligen Gewichten ist 1.0 extrem unwahrscheinlich
        assertTrue(cos < 0.999, "expected cosine < 0.999 for different inputs, got " + cos);
    }

    // ── helpers ──────────────────────────────────────────────────────────

    private static float[] rand(Random r, int n, float scale) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float) (r.nextGaussian() * scale);
        return a;
    }

    private static float[] zeros(int n) {
        return new float[n];
    }

    private static float[] ones(int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = 1f;
        return a;
    }
}

