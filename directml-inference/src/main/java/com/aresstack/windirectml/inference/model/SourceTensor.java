package com.aresstack.windirectml.inference.model;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * Format-neutral source tensor used before a model family compiles weights into
 * the internal runtime package layout.
 *
 * <p>Keep this class small: it describes a tensor, its storage kind, and an
 * optional read-only payload slice. It deliberately does not expose ONNX,
 * SafeTensors, or Hugging Face container types.</p>
 */
public record SourceTensor(
        String name,
        SourceTensorDataType dataType,
        long[] dims,
        TensorStorageKind storageKind,
        long byteLength,
        ByteBuffer payload
) {
    public SourceTensor {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("name must not be blank");
        }
        dataType = Objects.requireNonNull(dataType, "dataType");
        storageKind = Objects.requireNonNull(storageKind, "storageKind");
        dims = dims == null ? new long[0] : dims.clone();
        if (byteLength < 0) {
            throw new IllegalArgumentException("byteLength must be >= 0");
        }
        payload = readOnlyPayload(payload, byteLength);
    }

    public static SourceTensor inline(String name,
                                      int onnxDataType,
                                      long[] dims,
                                      long byteLength,
                                      ByteBuffer payload) {
        return new SourceTensor(name, SourceTensorDataType.fromOnnxCode(onnxDataType), dims,
                TensorStorageKind.INLINE, byteLength, payload);
    }

    public static SourceTensor inline(String name,
                                      SourceTensorDataType dataType,
                                      long[] dims,
                                      long byteLength,
                                      ByteBuffer payload) {
        return new SourceTensor(name, dataType, dims, TensorStorageKind.INLINE, byteLength, payload);
    }

    public static SourceTensor external(String name, int onnxDataType, long[] dims, long byteLength) {
        return new SourceTensor(name, SourceTensorDataType.fromOnnxCode(onnxDataType), dims,
                TensorStorageKind.EXTERNAL, byteLength, null);
    }

    public static SourceTensor metadataOnly(String name, int onnxDataType, long[] dims) {
        return new SourceTensor(name, SourceTensorDataType.fromOnnxCode(onnxDataType), dims,
                TensorStorageKind.METADATA_ONLY, 0L, null);
    }

    @Override
    public long[] dims() {
        return dims.clone();
    }

    public int onnxDataType() {
        return dataType.onnxCode();
    }

    public String dataTypeName() {
        return dataType.name();
    }

    public boolean hasPayload() {
        return payload != null && byteLength > 0;
    }

    public ByteBuffer payloadBuffer() {
        if (payload == null) {
            return ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN).asReadOnlyBuffer();
        }
        ByteBuffer duplicate = payload.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
        duplicate.position(0);
        duplicate.limit(Math.toIntExact(byteLength));
        return duplicate;
    }

    public TensorEntry toTensorEntry() {
        return new TensorEntry(name, onnxDataType(), dims, storageKind, byteLength);
    }

    public String shape() {
        return Arrays.toString(dims);
    }

    private static ByteBuffer readOnlyPayload(ByteBuffer payload, long byteLength) {
        if (payload == null || byteLength <= 0) {
            return null;
        }
        if (byteLength > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("payload byteLength exceeds current ByteBuffer-backed source tensor limit");
        }
        ByteBuffer duplicate = payload.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
        duplicate.position(0);
        duplicate.limit(Math.toIntExact(byteLength));
        return duplicate.slice().order(ByteOrder.LITTLE_ENDIAN).asReadOnlyBuffer();
    }
}
