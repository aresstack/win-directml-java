package com.aresstack.windirectml.inference.decoderonly;

import java.util.Objects;

/**
 * Contiguous key/value cache for the decoder-only WARP path.
 *
 * <p>One {@link DecoderOnlyWarpLayerKvCache} per hidden layer, each backed by pre-sized contiguous {@code float[]}
 * buffers. The capacity is bounded by the maximum sequence length of a single generation request, so the up-front
 * allocation stays proportional to the actual run instead of {@code max_position_embeddings}.</p>
 *
 * <p>Family-neutral: it depends only on the shared {@link DecoderOnlyConfig} shape contract (GQA head count, head
 * dim, layer count, max positions), so Qwen-like and Llama-like (SmolLM2) runtimes share it unchanged.</p>
 */
public final class DecoderOnlyWarpKvCache {

    private final DecoderOnlyWarpLayerKvCache[] layers;

    private DecoderOnlyWarpKvCache(DecoderOnlyWarpLayerKvCache[] layers) {
        this.layers = layers;
    }

    public static DecoderOnlyWarpKvCache create(DecoderOnlyConfig config, int maxTokens) {
        Objects.requireNonNull(config, "config");
        if (maxTokens <= 0) {
            throw new IllegalArgumentException("maxTokens must be > 0");
        }
        int keyWidth = Math.multiplyExact(config.numKeyValueHeads(), config.headDim());
        int capacity = Math.min(maxTokens, config.maxPositionEmbeddings());
        DecoderOnlyWarpLayerKvCache[] layerCaches = new DecoderOnlyWarpLayerKvCache[config.numHiddenLayers()];
        for (int layer = 0; layer < layerCaches.length; layer++) {
            layerCaches[layer] = new DecoderOnlyWarpLayerKvCache(keyWidth, capacity);
        }
        return new DecoderOnlyWarpKvCache(layerCaches);
    }

    public DecoderOnlyWarpLayerKvCache layer(int layerIndex) {
        return layers[layerIndex];
    }

    public int layerCount() {
        return layers.length;
    }

    public int completedTokenCount() {
        if (layers.length == 0) {
            return 0;
        }
        int completed = layers[0].size();
        for (DecoderOnlyWarpLayerKvCache layer : layers) {
            completed = Math.min(completed, layer.size());
        }
        return completed;
    }

    public void requireReadyForPosition(int position) {
        for (int layer = 0; layer < layers.length; layer++) {
            int size = layers[layer].size();
            if (size != position) {
                throw new IllegalStateException("Decoder-only WARP KV cache layer " + layer
                        + " is at position " + size + " but expected " + position);
            }
        }
    }
}
