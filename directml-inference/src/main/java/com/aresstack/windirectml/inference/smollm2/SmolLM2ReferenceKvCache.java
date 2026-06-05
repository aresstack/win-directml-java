package com.aresstack.windirectml.inference.smollm2;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Reference key/value cache for autoregressive SmolLM2 decoding.
 */
public final class SmolLM2ReferenceKvCache {

    private final List<SmolLM2ReferenceLayerKvCache> layers;

    private SmolLM2ReferenceKvCache(List<SmolLM2ReferenceLayerKvCache> layers) {
        this.layers = List.copyOf(layers);
    }

    public static SmolLM2ReferenceKvCache create(SmolLM2Config config) {
        Objects.requireNonNull(config, "config");
        int keyWidth = Math.multiplyExact(config.effectiveKeyValueHeads(), config.effectiveHeadDim());
        List<SmolLM2ReferenceLayerKvCache> layerCaches = new ArrayList<>(config.numHiddenLayers());
        for (int layer = 0; layer < config.numHiddenLayers(); layer++) {
            layerCaches.add(new SmolLM2ReferenceLayerKvCache(keyWidth));
        }
        return new SmolLM2ReferenceKvCache(layerCaches);
    }

    public SmolLM2ReferenceLayerKvCache layer(int layerIndex) {
        return layers.get(layerIndex);
    }

    public int layerCount() {
        return layers.size();
    }

    public int completedTokenCount() {
        if (layers.isEmpty()) {
            return 0;
        }
        int completed = layers.get(0).size();
        for (SmolLM2ReferenceLayerKvCache layer : layers) {
            completed = Math.min(completed, layer.size());
        }
        return completed;
    }

    public void requireReadyForPosition(int position) {
        for (int layer = 0; layer < layers.size(); layer++) {
            int size = layers.get(layer).size();
            if (size != position) {
                throw new IllegalStateException("SmolLM2 KV cache layer " + layer
                        + " is at position " + size + " but expected " + position);
            }
        }
    }
}
