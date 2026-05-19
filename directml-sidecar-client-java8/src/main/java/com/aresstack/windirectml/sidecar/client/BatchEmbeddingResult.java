package com.aresstack.windirectml.sidecar.client;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Typed view on the {@code embedBatch} response.
 *
 * <p>Returns one {@code float[]} per input text in input order.
 */
public final class BatchEmbeddingResult {

    private final List<float[]> vectors;
    private final int dimension;
    private final String model;
    private final boolean normalized;
    private final int count;
    private final long elapsedMillis;
    private final String raw;

    public BatchEmbeddingResult(List<float[]> vectors, int dimension, String model,
                                boolean normalized, int count,
                                long elapsedMillis, String raw) {
        this.vectors = (vectors == null)
                ? Collections.<float[]>emptyList()
                : Collections.unmodifiableList(new ArrayList<float[]>(vectors));
        this.dimension = dimension;
        this.model = model;
        this.normalized = normalized;
        this.count = count;
        this.elapsedMillis = elapsedMillis;
        this.raw = raw;
    }

    public static BatchEmbeddingResult from(JsonNode result, long elapsedMillis, String raw) {
        if (result == null) {
            return new BatchEmbeddingResult(Collections.<float[]>emptyList(),
                    0, null, false, 0, elapsedMillis, raw);
        }
        JsonNode arr = result.get("vectors");
        List<float[]> vecs = new ArrayList<float[]>();
        int dim = 0;
        if (arr != null && arr.isArray()) {
            for (int i = 0; i < arr.size(); i++) {
                JsonNode v = arr.get(i);
                if (v == null || !v.isArray()) {
                    vecs.add(new float[0]);
                    continue;
                }
                float[] f = new float[v.size()];
                for (int j = 0; j < v.size(); j++) {
                    f[j] = (float) v.get(j).asDouble();
                }
                vecs.add(f);
                if (f.length > dim) dim = f.length;
            }
        }
        int dimension = result.has("dimension") ? result.get("dimension").asInt(dim) : dim;
        String model = result.has("model") && !result.get("model").isNull()
                ? result.get("model").asText() : null;
        boolean normalized = result.has("normalized") && result.get("normalized").asBoolean(false);
        int count = result.has("count") ? result.get("count").asInt(vecs.size()) : vecs.size();
        return new BatchEmbeddingResult(vecs, dimension, model, normalized, count,
                elapsedMillis, raw);
    }

    public List<float[]> getVectors() {
        return vectors;
    }

    public int getDimension() {
        return dimension;
    }

    public String getModel() {
        return model;
    }

    public boolean isNormalized() {
        return normalized;
    }

    public int getCount() {
        return count;
    }

    public long getElapsedMillis() {
        return elapsedMillis;
    }

    public String getRaw() {
        return raw;
    }
}

