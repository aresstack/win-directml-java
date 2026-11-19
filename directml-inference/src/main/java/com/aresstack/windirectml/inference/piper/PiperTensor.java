package com.aresstack.windirectml.inference.piper;

import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Hold one typed tensor for the native VITS reference interpreter.
 * Keep data representations explicit so shape and type errors cannot silently corrupt synthesis.
 */
public final class PiperTensor {

    public static final int FLOAT32 = 1;
    public static final int INT32 = 6;
    public static final int INT64 = 7;
    public static final int BOOL = 9;

    private final int dataType;
    private final int[] shape;
    private final float[] floatValues;
    private final long[] integerValues;
    private final boolean[] booleanValues;

    private PiperTensor(int dataType, int[] shape, float[] floats,
                        long[] integers, boolean[] booleans) {
        this.dataType = dataType;
        this.shape = shape.clone();
        this.floatValues = floats;
        this.integerValues = integers;
        this.booleanValues = booleans;
        int actual = floats != null ? floats.length : integers != null ? integers.length : booleans.length;
        if (actual != elementCount(shape)) {
            throw new IllegalArgumentException("Piper tensor shape and data length differ");
        }
    }

    public static PiperTensor floats(float[] values, int... shape) {
        return new PiperTensor(FLOAT32, shape, values.clone(), null, null);
    }

    public static PiperTensor integers(long[] values, int... shape) {
        return new PiperTensor(INT64, shape, null, values.clone(), null);
    }

    public static PiperTensor bools(boolean[] values, int... shape) {
        return new PiperTensor(BOOL, shape, null, null, values.clone());
    }

    public static PiperTensor fromOnnx(OnnxTensor source) throws IOException {
        int[] shape;
        try {
            shape = Arrays.stream(source.dims()).mapToInt(Math::toIntExact).toArray();
        } catch (ArithmeticException failure) {
            throw new IOException("Piper tensor dimensions exceed Java array limits: " + source.name(), failure);
        }
        int count = elementCount(shape);
        int type = source.dataType();
        if (type == FLOAT32) {
            if (source.data() != null && source.data().length == count) {
                return floats(source.data(), shape);
            }
            ByteBuffer raw = source.rawDataBuffer().order(ByteOrder.LITTLE_ENDIAN);
            requireBytes(raw, (long) count * Float.BYTES, source.name());
            float[] values = new float[count];
            for (int i = 0; i < count; i++) {
                values[i] = raw.getFloat();
            }
            return floats(values, shape);
        }
        if (type == INT64 || type == INT32) {
            ByteBuffer raw = source.rawDataBuffer().order(ByteOrder.LITTLE_ENDIAN);
            int bytes = type == INT64 ? Long.BYTES : Integer.BYTES;
            requireBytes(raw, (long) count * bytes, source.name());
            long[] values = new long[count];
            for (int i = 0; i < count; i++) {
                values[i] = type == INT64 ? raw.getLong() : raw.getInt();
            }
            return new PiperTensor(type, shape, null, values, null);
        }
        if (type == BOOL) {
            ByteBuffer raw = source.rawDataBuffer();
            requireBytes(raw, count, source.name());
            boolean[] values = new boolean[count];
            for (int i = 0; i < count; i++) {
                values[i] = raw.get() != 0;
            }
            return bools(values, shape);
        }
        throw new IOException("Piper tensor type " + type + " is not supported: " + source.name());
    }

    public int dataType() {
        return dataType;
    }

    public int[] shape() {
        return shape.clone();
    }

    public int rank() {
        return shape.length;
    }

    public int length() {
        return elementCount(shape);
    }

    public float[] floatData() {
        return requireFloat().clone();
    }

    public long[] integerData() {
        return requireInteger().clone();
    }

    public boolean[] booleanData() {
        return requireBoolean().clone();
    }

    float[] requireFloat() {
        if (dataType != FLOAT32) {
            throw new IllegalArgumentException("Expected FLOAT32 tensor but found " + dataType);
        }
        return floatValues;
    }

    long[] requireInteger() {
        if (dataType != INT32 && dataType != INT64) {
            throw new IllegalArgumentException("Expected integer tensor but found " + dataType);
        }
        return integerValues;
    }

    boolean[] requireBoolean() {
        if (dataType != BOOL) {
            throw new IllegalArgumentException("Expected BOOL tensor but found " + dataType);
        }
        return booleanValues;
    }

    static int elementCount(int[] shape) {
        long count = 1;
        for (int dimension : shape) {
            if (dimension < 0) {
                throw new IllegalArgumentException("Dynamic tensor dimension is not resolved");
            }
            count *= dimension;
            if (count > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Piper tensor exceeds Java array limits");
            }
        }
        return (int) count;
    }

    private static void requireBytes(ByteBuffer buffer, long expected, String tensorName) throws IOException {
        if (buffer.remaining() != expected) {
            throw new IOException("Invalid ONNX payload for " + tensorName
                    + ": expected " + expected + " bytes, got " + buffer.remaining());
        }
    }

    @Override
    public String toString() {
        return "PiperTensor[type=" + dataType + ", shape=" + Arrays.toString(shape) + "]";
    }
}
