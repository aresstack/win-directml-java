package com.aresstack.windirectml.inference.smollm2;

import java.util.Objects;

/**
 * Float32 reference weights for one SmolLM2 decoder layer.
 */
public final class SmolLM2ReferenceLayerWeights {

    private final int layerIndex;
    private final SmolLM2DenseTensor inputNorm;
    private final SmolLM2DenseTensor queryProjection;
    private final SmolLM2DenseTensor keyProjection;
    private final SmolLM2DenseTensor valueProjection;
    private final SmolLM2DenseTensor outputProjection;
    private final SmolLM2DenseTensor postAttentionNorm;
    private final SmolLM2DenseTensor gateProjection;
    private final SmolLM2DenseTensor upProjection;
    private final SmolLM2DenseTensor downProjection;

    private SmolLM2ReferenceLayerWeights(int layerIndex,
                                         SmolLM2DenseTensor inputNorm,
                                         SmolLM2DenseTensor queryProjection,
                                         SmolLM2DenseTensor keyProjection,
                                         SmolLM2DenseTensor valueProjection,
                                         SmolLM2DenseTensor outputProjection,
                                         SmolLM2DenseTensor postAttentionNorm,
                                         SmolLM2DenseTensor gateProjection,
                                         SmolLM2DenseTensor upProjection,
                                         SmolLM2DenseTensor downProjection) {
        this.layerIndex = layerIndex;
        this.inputNorm = Objects.requireNonNull(inputNorm, "inputNorm");
        this.queryProjection = Objects.requireNonNull(queryProjection, "queryProjection");
        this.keyProjection = Objects.requireNonNull(keyProjection, "keyProjection");
        this.valueProjection = Objects.requireNonNull(valueProjection, "valueProjection");
        this.outputProjection = Objects.requireNonNull(outputProjection, "outputProjection");
        this.postAttentionNorm = Objects.requireNonNull(postAttentionNorm, "postAttentionNorm");
        this.gateProjection = Objects.requireNonNull(gateProjection, "gateProjection");
        this.upProjection = Objects.requireNonNull(upProjection, "upProjection");
        this.downProjection = Objects.requireNonNull(downProjection, "downProjection");
    }

    public static SmolLM2ReferenceLayerWeights from(SmolLM2LayerWeights weights) {
        Objects.requireNonNull(weights, "weights");
        return new SmolLM2ReferenceLayerWeights(
                weights.layerIndex(),
                SmolLM2DenseTensor.from(weights.require(SmolLM2TensorRole.LAYER_INPUT_NORM)),
                SmolLM2DenseTensor.from(weights.require(SmolLM2TensorRole.LAYER_SELF_Q)),
                SmolLM2DenseTensor.from(weights.require(SmolLM2TensorRole.LAYER_SELF_K)),
                SmolLM2DenseTensor.from(weights.require(SmolLM2TensorRole.LAYER_SELF_V)),
                SmolLM2DenseTensor.from(weights.require(SmolLM2TensorRole.LAYER_SELF_O)),
                SmolLM2DenseTensor.from(weights.require(SmolLM2TensorRole.LAYER_POST_ATTENTION_NORM)),
                SmolLM2DenseTensor.from(weights.require(SmolLM2TensorRole.LAYER_MLP_GATE)),
                SmolLM2DenseTensor.from(weights.require(SmolLM2TensorRole.LAYER_MLP_UP)),
                SmolLM2DenseTensor.from(weights.require(SmolLM2TensorRole.LAYER_MLP_DOWN)));
    }

    public int layerIndex() {
        return layerIndex;
    }

    public SmolLM2DenseTensor inputNorm() {
        return inputNorm;
    }

    public SmolLM2DenseTensor queryProjection() {
        return queryProjection;
    }

    public SmolLM2DenseTensor keyProjection() {
        return keyProjection;
    }

    public SmolLM2DenseTensor valueProjection() {
        return valueProjection;
    }

    public SmolLM2DenseTensor outputProjection() {
        return outputProjection;
    }

    public SmolLM2DenseTensor postAttentionNorm() {
        return postAttentionNorm;
    }

    public SmolLM2DenseTensor gateProjection() {
        return gateProjection;
    }

    public SmolLM2DenseTensor upProjection() {
        return upProjection;
    }

    public SmolLM2DenseTensor downProjection() {
        return downProjection;
    }
}
