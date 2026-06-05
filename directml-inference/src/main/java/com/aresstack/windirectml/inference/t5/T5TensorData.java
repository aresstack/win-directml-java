package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.RuntimeTensor;
import com.aresstack.windirectml.windows.OnnxModelReader;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * Float32 reference view of one runtime tensor.
 *
 * <p>Use this type only in the T5 reference pipeline. The production WARP
 * pipeline should keep payloads on the native/DirectML side and avoid copying
 * large tensors into Java arrays.</p>
 */
final class T5TensorData {
    private final String name;
    private final long[] dims;
    private final float[] values;

    private T5TensorData(String name, long[] dims, float[] values) {
        this.name = Objects.requireNonNull(name, "name");
        this.dims = dims == null ? new long[0] : dims.clone();
        this.values = Objects.requireNonNull(values, "values");
    }

    static T5TensorData from(RuntimeTensor tensor) {
        Objects.requireNonNull(tensor, "tensor");
        long[] dims = tensor.dims();
        int expectedElements = Math.toIntExact(elementCount(dims));
        float[] values = readFloatValues(tensor, expectedElements);
        return new T5TensorData(tensor.name(), dims, values);
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
        return Arrays.copyOf(values, values.length);
    }

    float at(int index) {
        return values[index];
    }

    float at(int row, int column) {
        return values[row * dim(1) + column];
    }

    int elementCount() {
        return values.length;
    }

    private static float[] readFloatValues(RuntimeTensor tensor, int expectedElements) {
        if (!tensor.hasPayload()) {
            throw new IllegalArgumentException("Tensor has no payload: " + tensor.name());
        }
        ByteBuffer buffer = tensor.rawDataBuffer().order(ByteOrder.LITTLE_ENDIAN);
        if (tensor.dataType() == OnnxModelReader.ONNX_FLOAT) {
            return readFloat32(buffer, expectedElements, tensor.name());
        }
        if (tensor.dataType() == OnnxModelReader.ONNX_FLOAT16) {
            return readFloat16(buffer, expectedElements, tensor.name());
        }
        throw new IllegalArgumentException("Unsupported T5 reference tensor dtype for " + tensor.name()
                + ": " + tensor.dataType());
    }

    private static float[] readFloat32(ByteBuffer buffer, int expectedElements, String tensorName) {
        if (buffer.remaining() < expectedElements * Float.BYTES) {
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
