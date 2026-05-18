package com.aresstack.windirectml.encoder.util;

/**
 * Kleine Hilfsklasse für Cosine Similarity – Vergleich zweier Embeddings.
 * <p>
 * Wird von den Referenz-Tests (Issue 20) genutzt, da Float-Gleichheit zu
 * fragil ist.
 */
public final class CosineSimilarity {

    private CosineSimilarity() {}

    public static double compute(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("length mismatch: " + a.length + " vs " + b.length);
        }
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            na  += (double) a[i] * a[i];
            nb  += (double) b[i] * b[i];
        }
        double denom = Math.sqrt(na) * Math.sqrt(nb);
        return denom == 0.0 ? 0.0 : dot / denom;
    }
}

