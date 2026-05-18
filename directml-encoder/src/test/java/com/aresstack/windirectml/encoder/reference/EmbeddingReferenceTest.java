package com.aresstack.windirectml.encoder.reference;

import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.encoder.util.CosineSimilarity;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

/**
 * Referenztests für Embeddings (Issue 20).
 * <p>
 * Strategie:
 * <ol>
 *   <li>Definierte Mini-Korpora (drei Texte, davon zwei semantisch nah).</li>
 *   <li>Erwartete Referenz-Cosine-Ähnlichkeiten werden offline mit der
 *       offiziellen Python-Implementierung von {@code sentence-transformers}
 *       erzeugt und hier hartcodiert.</li>
 *   <li>Vergleich erfolgt über {@link CosineSimilarity}, nicht über
 *       exakte Float-Gleichheit. Toleranz: ±0.02 (begründet durch
 *       FP16 / DirectML-Rundungsfehler).</li>
 * </ol>
 * <p>
 * Aktiviert wird der Test, sobald ein lauffähiges {@link EmbeddingModel}
 * vorhanden ist. Bis dahin: {@code @Disabled}.
 */
@Disabled("Requires working MiniLM encoder runtime (issues 11–18 must land first).")
class EmbeddingReferenceTest {

    private static final String[] CORPUS = {
            "A cat sits on the mat.",
            "A feline rests on a rug.",
            "The price of oil dropped sharply."
    };

    /** Erwartete Cosine-Ähnlichkeit zwischen 0 und 1 für die ersten beiden Texte. */
    private static final double EXPECTED_SIMILAR_PAIR = 0.78;
    /** Erwartete Cosine-Ähnlichkeit zwischen erstem und drittem Text (unähnlich). */
    private static final double EXPECTED_UNRELATED_PAIR = 0.18;
    private static final double TOLERANCE = 0.02;

    private EmbeddingModel model; // wird per Fixture geliefert, sobald verfügbar

    @Test
    void similarPairScoresHigh() throws Exception {
        EmbeddingVector a = model.embed(EmbeddingRequest.of(CORPUS[0]));
        EmbeddingVector b = model.embed(EmbeddingRequest.of(CORPUS[1]));
        double sim = CosineSimilarity.compute(a.values(), b.values());
        org.junit.jupiter.api.Assertions.assertTrue(
                Math.abs(sim - EXPECTED_SIMILAR_PAIR) <= TOLERANCE,
                "similar-pair cosine out of tolerance: " + sim);
    }

    @Test
    void unrelatedPairScoresLow() throws Exception {
        EmbeddingVector a = model.embed(EmbeddingRequest.of(CORPUS[0]));
        EmbeddingVector c = model.embed(EmbeddingRequest.of(CORPUS[2]));
        double sim = CosineSimilarity.compute(a.values(), c.values());
        org.junit.jupiter.api.Assertions.assertTrue(
                Math.abs(sim - EXPECTED_UNRELATED_PAIR) <= TOLERANCE,
                "unrelated-pair cosine out of tolerance: " + sim);
    }
}

