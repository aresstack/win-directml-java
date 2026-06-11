package com.aresstack.windirectml.inference.smollm2;

import java.util.Objects;

/**
 * Contiguous key/value cache for the native WARP decoder.
 *
 * <p>Mirrors {@link SmolLM2ReferenceKvCache} (same per-layer key width, same completed-token / readiness semantics)
 * but each layer is backed by pre-sized contiguous {@code float[]} buffers rather than a growing list of per-token
 * rows. The capacity is bounded by the maximum sequence length of a single generation request, so the up-front
 * allocation stays proportional to the actual run instead of {@code max_position_embeddings}.</p>
 */
final class SmolLM2WarpKvCache {

    private final SmolLM2WarpLayerKvCache[] layers;

    private SmolLM2WarpKvCache(SmolLM2WarpLayerKvCache[] layers) {
        this.layers = layers;
    }

    static SmolLM2WarpKvCache create(SmolLM2Config config, int maxTokens) {
        Objects.requireNonNull(config, "config");
        if (maxTokens <= 0) {
            throw new IllegalArgumentException("maxTokens must be > 0");
        }
        int keyWidth = Math.multiplyExact(config.effectiveKeyValueHeads(), config.effectiveHeadDim());
        int capacity = Math.min(maxTokens, config.maxPositionEmbeddings());
        SmolLM2WarpLayerKvCache[] layerCaches = new SmolLM2WarpLayerKvCache[config.numHiddenLayers()];
        for (int layer = 0; layer < layerCaches.length; layer++) {
            layerCaches[layer] = new SmolLM2WarpLayerKvCache(keyWidth, capacity);
        }
        return new SmolLM2WarpKvCache(layerCaches);
    }

    SmolLM2WarpLayerKvCache layer(int layerIndex) {
        return layers[layerIndex];
    }

    int layerCount() {
        return layers.length;
    }

    int completedTokenCount() {
        if (layers.length == 0) {
            return 0;
        }
        int completed = layers[0].size();
        for (SmolLM2WarpLayerKvCache layer : layers) {
            completed = Math.min(completed, layer.size());
        }
        return completed;
    }

    void requireReadyForPosition(int position) {
        for (int layer = 0; layer < layers.length; layer++) {
            int size = layers[layer].size();
            if (size != position) {
                throw new IllegalStateException("SmolLM2 WARP KV cache layer " + layer
                        + " is at position " + size + " but expected " + position);
            }
        }
    }
}
