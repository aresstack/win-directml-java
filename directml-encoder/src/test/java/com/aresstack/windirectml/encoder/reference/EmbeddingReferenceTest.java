package com.aresstack.windirectml.encoder.reference;

import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.encoder.minilm.CpuMiniLmEncoder;
import com.aresstack.windirectml.encoder.util.CosineSimilarity;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Referenztests gegen das echte {@code sentence-transformers/all-MiniLM-L6-v2}.
 * <p>
 * Wird automatisch deaktiviert, falls das Modell lokal nicht vorhanden ist.
 * Erwarteter Pfad (relative zum Repository-Root):
 * <pre>
 *   model/all-MiniLM-L6-v2/model.safetensors
 *   model/all-MiniLM-L6-v2/tokenizer.json
 * </pre>
 * Override via System-Property {@code minilm.testModelDir}.
 *
 * <p>Strategie (Issue 20): Vergleich über Cosine Similarity statt
 * exakter Float-Gleichheit. Toleranzen sind konservativ und validieren
 * das Encoder-Verhalten qualitativ.
 */
@EnabledIf("modelPresent")
class EmbeddingReferenceTest {

    private static final String[] CORPUS = {
            "A cat sits on the mat.",
            "A feline rests on a rug.",
            "The price of oil dropped sharply."
    };

    private static EmbeddingModel model;

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
        Path dir = resolveModelDir();
        if (dir != null) {
            // Runtime loads only from the package now: convert the reference model first.
            com.aresstack.windirectml.encoder.pack.EncoderWdmlPack.compile(dir,
                    dir.resolve(com.aresstack.windirectml.encoder.pack.EncoderWdmlPack.ENCODER_PACKAGE_FILE),
                    com.aresstack.windirectml.encoder.pack.EncoderWdmlPack.FAMILY_ENCODER);
        }
        model = CpuMiniLmEncoder.load(dir);
    }

    @Test
    void outputDimensionIs384() throws Exception {
        EmbeddingVector v = model.embed(EmbeddingRequest.of("hello world"));
        assertEquals(384, v.dimension());
        for (float f : v.values()) assertTrue(Float.isFinite(f));
    }

    @Test
    void normalizedOutputHasUnitNorm() throws Exception {
        EmbeddingVector v = model.embed(EmbeddingRequest.of("normalize me"));
        double norm = 0;
        for (float f : v.values()) norm += (double) f * f;
        assertEquals(1.0, Math.sqrt(norm), 1e-4);
    }

    @Test
    void semanticallySimilarTextsScoreHigh() throws Exception {
        EmbeddingVector a = model.embed(EmbeddingRequest.of(CORPUS[0]));
        EmbeddingVector b = model.embed(EmbeddingRequest.of(CORPUS[1]));
        double sim = CosineSimilarity.compute(a.values(), b.values());
        assertTrue(sim > 0.5, "expected similar pair cosine > 0.5, got " + sim);
    }

    @Test
    void unrelatedTextsScoreLow() throws Exception {
        EmbeddingVector a = model.embed(EmbeddingRequest.of(CORPUS[0]));
        EmbeddingVector c = model.embed(EmbeddingRequest.of(CORPUS[2]));
        double sim = CosineSimilarity.compute(a.values(), c.values());
        assertTrue(sim < 0.5, "expected unrelated pair cosine < 0.5, got " + sim);
    }

    @Test
    void similarPairCloserThanUnrelatedPair() throws Exception {
        EmbeddingVector a = model.embed(EmbeddingRequest.of(CORPUS[0]));
        EmbeddingVector b = model.embed(EmbeddingRequest.of(CORPUS[1]));
        EmbeddingVector c = model.embed(EmbeddingRequest.of(CORPUS[2]));
        double similar = CosineSimilarity.compute(a.values(), b.values());
        double unrelated = CosineSimilarity.compute(a.values(), c.values());
        assertTrue(similar > unrelated + 0.2,
                "expected separation: similar=" + similar + " vs unrelated=" + unrelated);
    }
}
