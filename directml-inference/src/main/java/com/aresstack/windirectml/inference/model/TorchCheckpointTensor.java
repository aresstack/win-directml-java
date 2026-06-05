package com.aresstack.windirectml.inference.model;

import java.util.Arrays;
import java.util.Objects;

/**
 * Tensor metadata extracted from a PyTorch state-dict checkpoint without
 * executing Python pickle payloads.
 */
public record TorchCheckpointTensor(
        String name,
        SourceTensorDataType dataType,
        long[] shape,
        long[] stride,
        String storageKey,
        String storageEntryName,
        long storageOffset,
        long storageElements,
        long storageByteLength,
        long tensorByteLength,
        boolean storageEntryPresent
) {
    public TorchCheckpointTensor {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("name must not be blank");
        }
        dataType = Objects.requireNonNull(dataType, "dataType");
        shape = shape == null ? new long[0] : shape.clone();
        stride = stride == null ? new long[0] : stride.clone();
        storageKey = storageKey == null ? "" : storageKey;
        storageEntryName = storageEntryName == null ? "" : storageEntryName;
        if (storageOffset < 0) {
            throw new IllegalArgumentException("storageOffset must be >= 0");
        }
        if (storageElements < 0) {
            throw new IllegalArgumentException("storageElements must be >= 0");
        }
        if (storageByteLength < 0) {
            throw new IllegalArgumentException("storageByteLength must be >= 0");
        }
        if (tensorByteLength < 0) {
            throw new IllegalArgumentException("tensorByteLength must be >= 0");
        }
    }

    @Override
    public long[] shape() {
        return shape.clone();
    }

    @Override
    public long[] stride() {
        return stride.clone();
    }

    public String shapeText() {
        return Arrays.toString(shape);
    }

    public String strideText() {
        return Arrays.toString(stride);
    }
}
