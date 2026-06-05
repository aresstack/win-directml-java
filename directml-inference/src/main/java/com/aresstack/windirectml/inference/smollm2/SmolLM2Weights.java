package com.aresstack.windirectml.inference.smollm2;

import java.util.List;
import java.util.Objects;

/**
 * Runtime-loadable SmolLM2 weights resolved from a wdmlpack payload.
 */
public final class SmolLM2Weights {

    private final SmolLM2Config config;
    private final SmolLM2WeightTensor tokenEmbedding;
    private final SmolLM2WeightTensor finalNorm;
    private final SmolLM2WeightTensor lmHead;
    private final boolean lmHeadTiedToEmbedding;
    private final List<SmolLM2LayerWeights> layers;
    private final long payloadBytes;

    public SmolLM2Weights(SmolLM2Config config,
                          SmolLM2WeightTensor tokenEmbedding,
                          SmolLM2WeightTensor finalNorm,
                          SmolLM2WeightTensor lmHead,
                          boolean lmHeadTiedToEmbedding,
                          List<SmolLM2LayerWeights> layers,
                          long payloadBytes) {
        this.config = Objects.requireNonNull(config, "config");
        this.tokenEmbedding = Objects.requireNonNull(tokenEmbedding, "tokenEmbedding");
        this.finalNorm = Objects.requireNonNull(finalNorm, "finalNorm");
        this.lmHead = Objects.requireNonNull(lmHead, "lmHead");
        this.lmHeadTiedToEmbedding = lmHeadTiedToEmbedding;
        this.layers = List.copyOf(Objects.requireNonNull(layers, "layers"));
        this.payloadBytes = payloadBytes;
    }

    public SmolLM2Config config() {
        return config;
    }

    public SmolLM2WeightTensor tokenEmbedding() {
        return tokenEmbedding;
    }

    public SmolLM2WeightTensor finalNorm() {
        return finalNorm;
    }

    public SmolLM2WeightTensor lmHead() {
        return lmHead;
    }

    public boolean lmHeadTiedToEmbedding() {
        return lmHeadTiedToEmbedding;
    }

    public List<SmolLM2LayerWeights> layers() {
        return layers;
    }

    public SmolLM2LayerWeights layer(int index) {
        if (index < 0 || index >= layers.size()) {
            throw new IndexOutOfBoundsException("SmolLM2 layer out of range: " + index);
        }
        return layers.get(index);
    }

    public long payloadBytes() {
        return payloadBytes;
    }
}
