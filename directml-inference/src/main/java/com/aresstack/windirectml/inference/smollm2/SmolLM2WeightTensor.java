package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.RuntimeTensor;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Objects;

/**
 * Runtime tensor bound to a SmolLM2 semantic role.
 */
public final class SmolLM2WeightTensor {

    private final SmolLM2TensorRoleBinding binding;
    private final RuntimeTensor tensor;

    public SmolLM2WeightTensor(SmolLM2TensorRoleBinding binding, RuntimeTensor tensor) {
        this.binding = Objects.requireNonNull(binding, "binding");
        this.tensor = Objects.requireNonNull(tensor, "tensor");
    }

    public SmolLM2TensorRoleBinding binding() {
        return binding;
    }

    public SmolLM2TensorRole role() {
        return binding.role();
    }

    public int layerIndex() {
        return binding.layerIndex();
    }

    public String tensorName() {
        return binding.tensorName();
    }

    public long[] dims() {
        return tensor.dims();
    }

    public int dataType() {
        return tensor.dataType();
    }

    public int rawByteLength() {
        return tensor.rawByteLength();
    }

    public ByteBuffer rawDataBuffer() {
        return tensor.rawDataBuffer();
    }

    public boolean hasShape(long... expected) {
        return Arrays.equals(expected, dims());
    }
}
