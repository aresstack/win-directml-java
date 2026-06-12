package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.RuntimeTensor;
import com.aresstack.windirectml.windows.OnnxModelReader;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * Float32 view of one runtime tensor.
 *
 * <p>Use this type only in the T5 reference pipeline and as the upload source for the T5 WARP
 * projections. The production WARP pipeline keeps payloads on the native/DirectML side and avoids
 * copying large tensors into Java arrays where possible.</p>
 *
 * <p><b>Heap-light FP32 path (slice H2b):</b> when constructed from a FLOAT32 {@link RuntimeTensor},
 * the float decode is <em>deferred</em>: the raw little-endian mmap {@link ByteBuffer} slice is retained
 * via {@link #fp32LittleEndianSource()} so the T5 WARP linear/LM-head adapters can upload it directly
 * (no host {@code float[]}). The {@code float[]} is only materialised on first {@link #values()} /
 * {@link #at(int)} access (reference math, fused projections, layernorm/bias/embedding), preserving the
 * existing {@code float[]} fallback. Validation (payload present, supported dtype, payload length) stays
 * eager, so error behaviour is unchanged. FLOAT16 and {@link #reference} tensors keep eager {@code float[]}
 * values and expose no FP32 source buffer.</p>
 */
final class T5TensorData {
    private final String name;
    private final long[] dims;
    private final int elementCount;
    private final ByteBuffer fp32SourceLe;   // nullable: present only for FP32 mmap-backed tensors
    private volatile float[] values;         // lazily decoded for fp32SourceLe; eager for FP16/reference

    private T5TensorData(String name, long[] dims, int elementCount, ByteBuffer fp32SourceLe, float[] values) {
        this.name = Objects.requireNonNull(name, "name");
        this.dims = dims == null ? new long[0] : dims.clone();
        this.elementCount = elementCount;
        this.fp32SourceLe = fp32SourceLe;
        this.values = values;
    }

    static T5TensorData from(RuntimeTensor tensor) {
        Objects.requireNonNull(tensor, "tensor");
        if (!tensor.hasPayload()) {
            throw new IllegalArgumentException("Tensor has no payload: " + tensor.name());
        }
        long[] dims = tensor.dims();
        int expectedElements = Math.toIntExact(elementCount(dims));
        ByteBuffer buffer = tensor.rawDataBuffer().order(ByteOrder.LITTLE_ENDIAN);
        if (tensor.dataType() == OnnxModelReader.ONNX_FLOAT) {
            // Eager length validation (same throw as before); decode is deferred to first values()/at().
            if (buffer.remaining() < (long) expectedElements * Float.BYTES) {
                throw new IllegalArgumentException("Tensor payload too short for FLOAT tensor: " + tensor.name());
            }
            return new T5TensorData(tensor.name(), dims, expectedElements, buffer, null);
        }
        if (tensor.dataType() == OnnxModelReader.ONNX_FLOAT16) {
            float[] values = readFloat16(buffer, expectedElements, tensor.name());
            return new T5TensorData(tensor.name(), dims, expectedElements, null, values);
        }
        throw new IllegalArgumentException("Unsupported T5 reference tensor dtype for " + tensor.name()
                + ": " + tensor.dataType());
    }

    static T5TensorData reference(String name, long[] dims, float[] values) {
        Objects.requireNonNull(values, "values");
        return new T5TensorData(name, dims, values.length, null, values.clone());
    }

    String name() {
        return name;
    }

    long[] dims() {
        return dims.clone();
    }

    int dim(int index) {
        return Math.toIntExact(dims[index]);
    }

    int rank() {
        return dims.length;
    }

    float[] values() {
        float[] decoded = ensureValues();
        return Arrays.copyOf(decoded, decoded.length);
    }

    float at(int index) {
        return ensureValues()[index];
    }

    float at(int row, int column) {
        return ensureValues()[row * dim(1) + column];
    }

    int elementCount() {
        return elementCount;
    }

    /**
     * The raw little-endian FP32 payload slice for direct (heap-light) WARP upload, or {@code null} when this
     * tensor is not a plain FLOAT32 mmap-backed tensor (FLOAT16 or a fused/reference tensor). The returned buffer
     * is an independent view (own position/limit) over the shared mmap region; the caller must not rely on the
     * mapping outliving the runtime tensor catalog.
     */
    ByteBuffer fp32LittleEndianSource() {
        return fp32SourceLe == null ? null : fp32SourceLe.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    }

    private float[] ensureValues() {
        float[] v = values;
        if (v == null) {
            // Idempotent lazy decode (load-time / single-threaded use); publication via the volatile field.
            v = readFloat32(fp32SourceLe.duplicate().order(ByteOrder.LITTLE_ENDIAN), elementCount, name);
            values = v;
        }
        return v;
    }

    private static float[] readFloat32(ByteBuffer buffer, int expectedElements, String tensorName) {
        if (buffer.remaining() < (long) expectedElements * Float.BYTES) {
            throw new IllegalArgumentException("Tensor payload too short for FLOAT tensor: " + tensorName);
        }
        float[] result = new float[expectedElements];
        for (int i = 0; i < result.length; i++) {
            result[i] = buffer.getFloat();
        }
        return result;
    }

    private static float[] readFloat16(ByteBuffer buffer, int expectedElements, String tensorName) {
        if (buffer.remaining() < expectedElements * Short.BYTES) {
            throw new IllegalArgumentException("Tensor payload too short for FLOAT16 tensor: " + tensorName);
        }
        float[] result = new float[expectedElements];
        for (int i = 0; i < result.length; i++) {
            result[i] = fp16ToFloat(buffer.getShort());
        }
        return result;
    }

    private static long elementCount(long[] dims) {
        long total = 1;
        for (long dim : dims) {
            if (dim <= 0) {
                throw new IllegalArgumentException("Tensor dimensions must be positive: " + Arrays.toString(dims));
            }
            total = Math.multiplyExact(total, dim);
        }
        return total;
    }

    private static float fp16ToFloat(short half) {
        int h = half & 0xffff;
        int sign = (h >>> 15) & 0x00000001;
        int exponent = (h >>> 10) & 0x0000001f;
        int fraction = h & 0x000003ff;
        int bits;
        if (exponent == 0) {
            if (fraction == 0) {
                bits = sign << 31;
            } else {
                while ((fraction & 0x00000400) == 0) {
                    fraction <<= 1;
                    exponent--;
                }
                exponent++;
                fraction &= ~0x00000400;
                bits = (sign << 31) | ((exponent + (127 - 15)) << 23) | (fraction << 13);
            }
        } else if (exponent == 31) {
            bits = (sign << 31) | 0x7f800000 | (fraction << 13);
        } else {
            bits = (sign << 31) | ((exponent + (127 - 15)) << 23) | (fraction << 13);
        }
        return Float.intBitsToFloat(bits);
    }
}
