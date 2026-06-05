package com.aresstack.windirectml.inference.model;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Runtime-side tensor directory built from a {@code .wdmlpack} payload.
 *
 * <p>The catalog owns no model-family semantics. It simply provides named
 * tensor access and adapters back to the current import seam while the Qwen
 * runtime is migrated away from ONNX-shaped weight objects.</p>
 */
public final class RuntimeTensorCatalog {

    private final Map<String, RuntimeTensor> tensors;

    public RuntimeTensorCatalog(Collection<RuntimeTensor> tensors) {
        Objects.requireNonNull(tensors, "tensors");
        LinkedHashMap<String, RuntimeTensor> byName = new LinkedHashMap<>();
        for (RuntimeTensor tensor : tensors) {
            if (tensor != null && tensor.name() != null && !tensor.name().isBlank()) {
                byName.put(tensor.name(), tensor);
            }
        }
        this.tensors = Map.copyOf(byName);
    }

    public static RuntimeTensorCatalog empty() {
        return new RuntimeTensorCatalog(java.util.List.of());
    }

    public RuntimeTensor get(String name) {
        return tensors.get(name);
    }

    public boolean contains(String name) {
        return tensors.containsKey(name);
    }

    public Collection<RuntimeTensor> values() {
        return tensors.values();
    }

    public Map<String, RuntimeTensor> asMap() {
        return tensors;
    }

    public int size() {
        return tensors.size();
    }

    public boolean isEmpty() {
        return tensors.isEmpty();
    }

    public long payloadBytes() {
        return tensors.values().stream()
                .filter(RuntimeTensor::hasPayload)
                .mapToLong(RuntimeTensor::rawByteLength)
                .sum();
    }

    public SourceTensorCatalog toSourceTensorCatalog() {
        return new SourceTensorCatalog(tensors.values().stream()
                .map(tensor -> SourceTensor.inline(tensor.name(), tensor.dataType(), tensor.dims(),
                        tensor.rawByteLength(), tensor.hasPayload() ? tensor.rawDataBuffer() : null))
                .toList());
    }

    public TensorCatalog toTensorCatalog() {
        return toSourceTensorCatalog().toTensorCatalog();
    }
}
