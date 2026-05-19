package com.aresstack.windirectml.sidecar.client;

import com.fasterxml.jackson.databind.JsonNode;

/**
 * Typed view on the {@code embed} response.
 */
public final class EmbeddingResult {

    private final float[] vector;
    private final int dimension;
    private final String model;
    private final boolean normalized;
    private final long elapsedMillis;
    private final String raw;

    public EmbeddingResult(float[] vector, int dimension, String model,
                           boolean normalized, long elapsedMillis, String raw) {
        this.vector = vector;
        this.dimension = dimension;
        this.model = model;
        this.normalized = normalized;
        this.elapsedMillis = elapsedMillis;
        this.raw = raw;
    }

    public static EmbeddingResult from(JsonNode result, long elapsedMillis, String raw) {
        if (result == null) {
            return new EmbeddingResult(new float[0], 0, null, false, elapsedMillis, raw);
        }
        JsonNode vec = result.get("vector");
        float[] values;
        if (vec != null && vec.isArray()) {
            values = new float[vec.size()];
            for (int i = 0; i < vec.size(); i++) {
                values[i] = (float) vec.get(i).asDouble();
            }
        } else {
            values = new float[0];
        }
        int dim = result.has("dimension") ? result.get("dimension").asInt() : values.length;
        String model = result.has("model") && !result.get("model").isNull()
                ? result.get("model").asText() : null;
        boolean normalized = result.has("normalized") && result.get("normalized").asBoolean(false);
        return new EmbeddingResult(values, dim, model, normalized, elapsedMillis, raw);
    }

    public float[] getVector()     { return vector; }
    public int getDimension()      { return dimension; }
    public String getModel()       { return model; }
    public boolean isNormalized()  { return normalized; }
    public long getElapsedMillis() { return elapsedMillis; }
    public String getRaw()         { return raw; }

    /**
     * Compute cosine similarity between two embedding vectors.
     * Returns 0 if either vector is empty or all-zero.
     */
    public static double cosine(float[] a, float[] b) {
        if (a == null || b == null || a.length == 0 || b.length == 0) return 0.0;
        int n = Math.min(a.length, b.length);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < n; i++) {
            dot += (double) a[i] * b[i];
            na  += (double) a[i] * a[i];
            nb  += (double) b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0.0;
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
}

