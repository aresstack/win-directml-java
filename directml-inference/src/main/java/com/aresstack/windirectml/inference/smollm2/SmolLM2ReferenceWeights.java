package com.aresstack.windirectml.inference.smollm2;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Float32 reference view over SmolLM2 weights.
 */
public final class SmolLM2ReferenceWeights {

    private final SmolLM2Config config;
    private final SmolLM2DenseTensor tokenEmbedding;
    private final SmolLM2DenseTensor finalNorm;
    private final SmolLM2DenseTensor lmHead;
    private final boolean lmHeadTiedToEmbedding;
    private final List<SmolLM2ReferenceLayerWeights> layers;

    private SmolLM2ReferenceWeights(SmolLM2Config config,
                                    SmolLM2DenseTensor tokenEmbedding,
                                    SmolLM2DenseTensor finalNorm,
                                    SmolLM2DenseTensor lmHead,
                                    boolean lmHeadTiedToEmbedding,
                                    List<SmolLM2ReferenceLayerWeights> layers) {
        this.config = Objects.requireNonNull(config, "config");
        this.tokenEmbedding = Objects.requireNonNull(tokenEmbedding, "tokenEmbedding");
        this.finalNorm = Objects.requireNonNull(finalNorm, "finalNorm");
        this.lmHead = Objects.requireNonNull(lmHead, "lmHead");
        this.lmHeadTiedToEmbedding = lmHeadTiedToEmbedding;
        this.layers = List.copyOf(layers);
    }

    public static SmolLM2ReferenceWeights from(SmolLM2Weights weights) {
        Objects.requireNonNull(weights, "weights");
        List<SmolLM2ReferenceLayerWeights> layers = new ArrayList<>(weights.layers().size());
        for (SmolLM2LayerWeights layer : weights.layers()) {
            layers.add(SmolLM2ReferenceLayerWeights.from(layer));
        }
        return new SmolLM2ReferenceWeights(
                weights.config(),
                SmolLM2DenseTensor.from(weights.tokenEmbedding()),
                SmolLM2DenseTensor.from(weights.finalNorm()),
                SmolLM2DenseTensor.from(weights.lmHead()),
                weights.lmHeadTiedToEmbedding(),
                layers);
    }

    public SmolLM2Config config() {
        return config;
    }

    public SmolLM2DenseTensor tokenEmbedding() {
        return tokenEmbedding;
    }

    public SmolLM2DenseTensor finalNorm() {
        return finalNorm;
    }

    public SmolLM2DenseTensor lmHead() {
        return lmHead;
    }

    public boolean lmHeadTiedToEmbedding() {
        return lmHeadTiedToEmbedding;
    }

    public List<SmolLM2ReferenceLayerWeights> layers() {
        return layers;
    }
}
