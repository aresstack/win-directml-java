package com.aresstack.windirectml.inference.smollm2;

import java.util.Objects;

/**
 * Planned buffer entry with aligned placement information for native upload.
 */
public record SmolLM2WarpBufferEntry(String name,
                                     SmolLM2WarpBufferKind kind,
                                     long offset,
                                     long byteLength,
                                     long alignedByteLength,
                                     long[] dims,
                                     int dataType,
                                     SmolLM2TensorRole role,
                                     int layerIndex) {

    public SmolLM2WarpBufferEntry {
        name = Objects.requireNonNull(name, "name");
        kind = Objects.requireNonNull(kind, "kind");
        dims = dims == null ? new long[0] : dims.clone();
        if (offset < 0L) {
            throw new IllegalArgumentException("offset must not be negative");
        }
        if (byteLength < 0L) {
            throw new IllegalArgumentException("byteLength must not be negative");
        }
        if (alignedByteLength < byteLength) {
            throw new IllegalArgumentException("alignedByteLength must be >= byteLength");
        }
    }

    public long endOffset() {
        return offset + alignedByteLength;
    }

    @Override
    public long[] dims() {
        return dims.clone();
    }
}
