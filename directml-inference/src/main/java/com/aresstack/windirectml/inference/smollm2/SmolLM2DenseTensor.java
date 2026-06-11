package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyReferenceDenseOps;
import com.aresstack.windirectml.inference.model.SourceTensorDataType;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * Float32 view over a SmolLM2 runtime tensor payload.
 */
public final class SmolLM2DenseTensor {

    private final String name;
    private final long[] dims;
    private final float[] values;

    private SmolLM2DenseTensor(String name, long[] dims, float[] values) {
        this.name = Objects.requireNonNull(name, "name");
        this.dims = dims.clone();
        this.values = values.clone();
    }

    public static SmolLM2DenseTensor from(SmolLM2WeightTensor tensor) {
        Objects.requireNonNull(tensor, "tensor");
        long[] dims = tensor.dims();
        int elementCount = checkedElementCount(dims);
        return new SmolLM2DenseTensor(tensor.tensorName(), dims, decodeValues(tensor, elementCount));
    }

    public String name() {
        return name;
    }

    public long[] dims() {
        return dims.clone();
    }

    public int rank() {
        return dims.length;
    }

    public int dim(int index) {
        return Math.toIntExact(dims[index]);
    }

    public int elementCount() {
        return values.length;
    }

    public float value(int index) {
        return values[index];
    }

    public float[] copyValues() {
        return values.clone();
    }

    public float[] copyRow(int rowIndex) {
        requireRank(2);
        int rows = dim(0);
        int cols = dim(1);
        if (rowIndex < 0 || rowIndex >= rows) {
            throw new IllegalArgumentException("rowIndex out of range for " + name + ": " + rowIndex);
        }
        return Arrays.copyOfRange(values, rowIndex * cols, rowIndex * cols + cols);
    }

    /** Copy row {@code rowIndex} into {@code target[0..cols)} without allocating a new array. */
    public void copyRowInto(int rowIndex, float[] target) {
        requireRank(2);
        int rows = dim(0);
        int cols = dim(1);
        if (rowIndex < 0 || rowIndex >= rows) {
            throw new IllegalArgumentException("rowIndex out of range for " + name + ": " + rowIndex);
        }
        if (target.length < cols) {
            throw new IllegalArgumentException("target too small for row of " + name
                    + ": target=" + target.length + ", expected at least=" + cols);
        }
        System.arraycopy(values, rowIndex * cols, target, 0, cols);
    }

    public float[] multiplyVector(float[] input) {
        requireRank(2);
        Objects.requireNonNull(input, "input");
        int rows = dim(0);
        int cols = dim(1);
        if (input.length != cols) {
            throw new IllegalArgumentException("Input width mismatch for " + name
                    + ": expected " + cols + " but got " + input.length);
        }
        float[] output = new float[rows];
        DecoderOnlyReferenceDenseOps.multiplyRows(values, rows, cols, input, output);
        return output;
    }

    public float dotRow(int rowIndex, float[] input) {
        requireRank(2);
        Objects.requireNonNull(input, "input");
        int rows = dim(0);
        int cols = dim(1);
        if (rowIndex < 0 || rowIndex >= rows) {
            throw new IllegalArgumentException("rowIndex out of range for " + name + ": " + rowIndex);
        }
        if (input.length != cols) {
            throw new IllegalArgumentException("Input width mismatch for " + name
                    + ": expected " + cols + " but got " + input.length);
        }
        return DecoderOnlyReferenceDenseOps.dot(values, rowIndex * cols, input, 0, cols);
    }

    private void requireRank(int expectedRank) {
        if (dims.length != expectedRank) {
            throw new IllegalStateException(name + " must have rank " + expectedRank
                    + " but has " + dims.length);
        }
    }

    private static float[] decodeValues(SmolLM2WeightTensor tensor, int elementCount) {
        ByteBuffer raw = tensor.rawDataBuffer().order(ByteOrder.LITTLE_ENDIAN);
        SourceTensorDataType dataType = SourceTensorDataType.fromOnnxCode(tensor.dataType());
        int expectedBytes = Math.multiplyExact(elementCount, dataType.bytesPerElement());
        if (raw.remaining() < expectedBytes) {
            throw new IllegalStateException("Tensor payload is shorter than its shape requires: "
                    + tensor.tensorName());
        }
        float[] values = new float[elementCount];
        for (int i = 0; i < elementCount; i++) {
            values[i] = switch (dataType.name()) {
                case "FLOAT" -> raw.getFloat();
                case "FLOAT16" -> halfToFloat(raw.getShort());
                case "BFLOAT16" -> Float.intBitsToFloat((raw.getShort() & 0xFFFF) << 16);
                default -> throw new IllegalStateException("Unsupported SmolLM2 runtime dtype "
                        + dataType.name() + " for tensor " + tensor.tensorName());
            };
        }
        return values;
    }

    private static float halfToFloat(short half) {
        int bits = half & 0xFFFF;
        int sign = (bits & 0x8000) << 16;
        int exponent = (bits >>> 10) & 0x1F;
        int mantissa = bits & 0x03FF;
        if (exponent == 0) {
            if (mantissa == 0) {
                return Float.intBitsToFloat(sign);
            }
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            exponent++;
            mantissa &= ~0x0400;
        } else if (exponent == 0x1F) {
            return Float.intBitsToFloat(sign | 0x7F800000 | (mantissa << 13));
        }
        exponent = exponent + (127 - 15);
        mantissa = mantissa << 13;
        return Float.intBitsToFloat(sign | (exponent << 23) | mantissa);
    }

    private static int checkedElementCount(long[] dims) {
        long count = 1L;
        for (long dim : dims) {
            if (dim <= 0L) {
                throw new IllegalStateException("Tensor dimension must be positive: " + dim);
            }
            count = Math.multiplyExact(count, dim);
        }
        return Math.toIntExact(count);
    }
}
