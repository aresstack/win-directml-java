package com.aresstack.windirectml.inference.model;

import java.util.Arrays;

/**
 * Source-format-neutral tensor metadata used by the import layer.
 */
public record TensorEntry(
        String name,
        int dataType,
        long[] dims,
        TensorStorageKind storageKind,
        long byteLength
) {
    public TensorEntry {
        dims = dims != null ? dims.clone() : new long[0];
    }

    @Override
    public long[] dims() {
        return dims.clone();
    }

    public String shape() {
        return Arrays.toString(dims);
    }
}
