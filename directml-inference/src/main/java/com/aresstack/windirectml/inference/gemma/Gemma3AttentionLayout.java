package com.aresstack.windirectml.inference.gemma;

import java.util.Objects;

/**
 * Device-free description of Gemma 3 attention geometry for a given config: per-layer local/global
 * classification, the per-layer RoPE base, GQA grouping, and the sliding-window key range. This is the
 * shared layout the CPU reference and the (later) WARP attention step both consume so the masking and
 * head mapping stay identical across paths.
 */
public final class Gemma3AttentionLayout {

    private final Gemma3Config config;

    public Gemma3AttentionLayout(Gemma3Config config) {
        this.config = Objects.requireNonNull(config, "config");
    }

    public int numHeads() {
        return config.numAttentionHeads();
    }

    public int numKeyValueHeads() {
        return config.numKeyValueHeads();
    }

    public int headDim() {
        return config.headDim();
    }

    /** Number of query heads sharing each kv head (GQA group size). */
    public int groupsPerKvHead() {
        return config.numAttentionHeads() / config.numKeyValueHeads();
    }

    /** The kv head index serving query head {@code queryHead}. */
    public int kvHeadFor(int queryHead) {
        return queryHead / groupsPerKvHead();
    }

    public boolean isFullAttention(int layer) {
        return config.isFullAttentionLayer(layer);
    }

    public double ropeTheta(int layer) {
        return config.ropeThetaForLayer(layer);
    }

    public float attentionScale() {
        return (float) config.attentionScale();
    }

    /**
     * First (inclusive) key position attended by query position {@code queryPos} in {@code layer}:
     * 0 for global/full layers, {@code max(0, queryPos - sliding_window + 1)} for local layers.
     */
    public int firstValidKey(int layer, int queryPos) {
        if (isFullAttention(layer) || config.slidingWindow() <= 0) {
            return 0;
        }
        return Math.max(0, queryPos - config.slidingWindow() + 1);
    }

    /** Whether key position {@code keyPos} is visible to query position {@code queryPos} in {@code layer}. */
    public boolean isVisible(int layer, int queryPos, int keyPos) {
        return keyPos <= queryPos && keyPos >= firstValidKey(layer, queryPos); // causal + window
    }
}
