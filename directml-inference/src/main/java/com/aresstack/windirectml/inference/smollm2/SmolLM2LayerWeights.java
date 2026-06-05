package com.aresstack.windirectml.inference.smollm2;

import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;

/**
 * Role-indexed weights for one SmolLM2 decoder layer.
 */
public final class SmolLM2LayerWeights {

    private final int layerIndex;
    private final Map<SmolLM2TensorRole, SmolLM2WeightTensor> tensors;

    public SmolLM2LayerWeights(int layerIndex, Map<SmolLM2TensorRole, SmolLM2WeightTensor> tensors) {
        if (layerIndex < 0) {
            throw new IllegalArgumentException("layerIndex must be >= 0");
        }
        this.layerIndex = layerIndex;
        this.tensors = Map.copyOf(new EnumMap<>(Objects.requireNonNull(tensors, "tensors")));
    }

    public int layerIndex() {
        return layerIndex;
    }

    public SmolLM2WeightTensor require(SmolLM2TensorRole role) {
        SmolLM2WeightTensor tensor = tensors.get(role);
        if (tensor == null) {
            throw new IllegalStateException("Missing SmolLM2 layer tensor: " + role + "#" + layerIndex);
        }
        return tensor;
    }

    public Map<SmolLM2TensorRole, SmolLM2WeightTensor> tensors() {
        return tensors;
    }
}
