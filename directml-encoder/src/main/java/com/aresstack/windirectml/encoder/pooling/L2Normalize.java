package com.aresstack.windirectml.encoder.pooling;

/**
 * CPU-Referenzimplementierung der L2-Normalisierung.
 * <p>
 * Normalisiert {@code in-place} auf Einheitsnorm. Für sehr kleine Vektoren
 * wird {@code epsilon} eingesetzt, um Division durch Null zu vermeiden.
 */
public final class L2Normalize {

    private L2Normalize() {}

    public static void inPlace(float[] vector, float epsilon) {
        double sum = 0.0;
        for (float v : vector) sum += (double) v * v;
        double norm = Math.sqrt(sum);
        double inv = 1.0 / Math.max(norm, epsilon);
        for (int i = 0; i < vector.length; i++) vector[i] = (float) (vector[i] * inv);
    }

    public static float[] normalize(float[] vector, float epsilon) {
        float[] out = vector.clone();
        inPlace(out, epsilon);
        return out;
    }
}

