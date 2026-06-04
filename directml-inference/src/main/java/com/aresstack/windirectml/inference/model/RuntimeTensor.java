package com.aresstack.windirectml.inference.model;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * Tensor payload mapped from an internal runtime package.
 *
 * <p>This type deliberately contains no ONNX or SafeTensors concepts. Model
 * families may adapt it to their current weight structures, but the package
 * reader itself remains format-neutral.</p>
 */
public final class RuntimeTensor {

    private final String name;
    private final long[] dims;
    private final int dataType;
    private final ByteBuffer rawData;
    private final int rawByteLength;

    public RuntimeTensor(String name, long[] dims, int dataType, ByteBuffer rawData, int rawByteLength) {
        this.name = Objects.requireNonNull(name, "name");
        this.dims = dims == null ? new long[0] : dims.clone();
        this.dataType = dataType;
        this.rawData = rawData == null ? ByteBuffer.allocate(0).asReadOnlyBuffer() : rawData.asReadOnlyBuffer();
        this.rawByteLength = Math.max(0, rawByteLength);
    }

    public String name() {
        return name;
    }

    public long[] dims() {
        return dims.clone();
    }

    public int dataType() {
        return dataType;
    }

    public ByteBuffer rawDataBuffer() {
        ByteBuffer duplicate = rawData.asReadOnlyBuffer();
        duplicate.position(0);
        return duplicate;
    }

    public int rawByteLength() {
        return rawByteLength;
    }

    public boolean hasPayload() {
        return rawByteLength > 0;
    }
}
