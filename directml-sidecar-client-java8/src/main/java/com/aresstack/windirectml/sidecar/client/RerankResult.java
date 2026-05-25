package com.aresstack.windirectml.sidecar.client;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Typed view on the {@code rerank} response.
 * <p>
 * Carries the model name plus the ranked list of {@link Item items},
 * already sorted by descending score on the sidecar. The host re-maps
 * {@link Item#getIndex()} to the original document text on its side.
 *
 * <h3>Score semantics</h3>
 * <ul>
 *   <li><b>Model-dependent:</b> score values are raw classifier logits from the
 *       loaded cross-encoder model. Different models can produce different
 *       value ranges, so scores are not comparable across models.</li>
 *   <li><b>Intra-query only:</b> scores are intended for ranking documents
 *       within a single query. Comparing scores across different queries is not
 *       meaningful.</li>
 *   <li><b>Not globally calibrated:</b> scores are not probabilities and have no
 *       fixed threshold for "relevant" vs. "not relevant".</li>
 * </ul>
 *
 * <p>The items list is sorted <b>descending by score</b> (highest = most relevant first).
 */
public final class RerankResult {

    private final String model;
    private final List<Item> items;
    private final long elapsedMillis;
    private final String raw;

    public RerankResult(String model, List<Item> items, long elapsedMillis, String raw) {
        this.model = model;
        this.items = Collections.unmodifiableList(new ArrayList<Item>(items));
        this.elapsedMillis = elapsedMillis;
        this.raw = raw;
    }

    public static RerankResult from(JsonNode result, long elapsedMillis, String raw) {
        if (result == null) {
            return new RerankResult(null, Collections.<Item>emptyList(), elapsedMillis, raw);
        }
        String model = result.has("model") && !result.get("model").isNull()
                ? result.get("model").asText() : null;
        List<Item> items = new ArrayList<Item>();
        JsonNode arr = result.get("results");
        if (arr != null && arr.isArray()) {
            for (int i = 0; i < arr.size(); i++) {
                JsonNode r = arr.get(i);
                int index = r.has("index") ? r.get("index").asInt(-1) : -1;
                double score = r.has("score") ? r.get("score").asDouble(0.0) : 0.0;
                items.add(new Item(index, score));
            }
        }
        return new RerankResult(model, items, elapsedMillis, raw);
    }

    public String getModel() { return model; }
    public List<Item> getItems() { return items; }
    public long getElapsedMillis() { return elapsedMillis; }
    public String getRaw() { return raw; }

    /**
     * One reranked entry: original document index plus raw classifier logit.
     * <p>
     * The {@link #getScore() score} is model-dependent and only meaningful for
     * relative ranking within the same query. It is not a probability and cannot
     * be compared across different models or queries.
     */
    public static final class Item {
        private final int index;
        private final double score;

        public Item(int index, double score) {
            this.index = index;
            this.score = score;
        }

        public int getIndex() { return index; }
        public double getScore() { return score; }
    }
}

