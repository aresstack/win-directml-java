package com.aresstack.windirectml.inference.whisper;

import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorEntry;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorsFile;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Load Hugging Face Whisper SafeTensors weights without ONNX Runtime, Torch,
 * Python, JNI or custom DLLs. Decode individual tensors lazily into float32.
 */
public final class WhisperCheckpoint {
    private final SafeTensorsFile file;
    private final Map<String, Tensor> loaded = new LinkedHashMap<String, Tensor>();

    private WhisperCheckpoint(SafeTensorsFile file) {
        this.file = Objects.requireNonNull(file, "file");
    }

    public static WhisperCheckpoint open(Path safeTensorsFile) throws IOException {
        return new WhisperCheckpoint(SafeTensorsReader.read(safeTensorsFile));
    }

    public int tensorCount() {
        return file.tensors().size();
    }

    public synchronized Tensor require(String name) throws IOException {
        Tensor tensor = loaded.get(name);
        if (tensor != null) {
            return tensor;
        }
        SafeTensorEntry entry = file.tensors().get(name);
        if (entry == null) {
            throw new IOException("Missing Whisper tensor: " + name);
        }
        tensor = decode(entry);
        loaded.put(name, tensor);
        return tensor;
    }

    private static Tensor decode(SafeTensorEntry entry) throws IOException {
        long[] originalShape = entry.shape();
        int[] dimensions = new int[originalShape.length];
        long count = 1;
        for (int i = 0; i < originalShape.length; i++) {
            if (originalShape[i] <= 0 || originalShape[i] > Integer.MAX_VALUE) {
                throw new IOException("Unsupported Whisper dimension: " + entry.name());
            }
            dimensions[i] = (int) originalShape[i];
            count = Math.multiplyExact(count, originalShape[i]);
        }
        if (count > Integer.MAX_VALUE) {
            throw new IOException("Tensor too large: " + entry.name());
        }
        float[] values = new float[(int) count];
        ByteBuffer data = entry.dataBuffer();
        switch (entry.dtype()) {
            case "F32":
                for (int i = 0; i < values.length; i++) {
                    values[i] = data.getFloat(i * 4);
                }
                break;
            case "F16":
                for (int i = 0; i < values.length; i++) {
                    values[i] = decodeHalf(data.getShort(i * 2));
                }
                break;
            case "BF16":
                for (int i = 0; i < values.length; i++) {
                    values[i] = Float.intBitsToFloat((data.getShort(i * 2) & 0xffff) << 16);
                }
                break;
            default:
                throw new IOException("Whisper requires float weights, got " + entry.dtype()
                        + " for tensor " + entry.name());
        }
        return new Tensor(dimensions, values);
    }

    private static float decodeHalf(short encoded) {
        int bits = encoded & 0xffff;
        int exponent = (bits >>> 10) & 0x1f;
        int mantissa = bits & 0x3ff;
        int sign = (bits & 0x8000) << 16;
        int singleBits;
        if (exponent == 0) {
            if (mantissa == 0) {
                singleBits = sign;
            } else {
                int shift = Integer.numberOfLeadingZeros(mantissa) - 21;
                int normalized = mantissa << shift;
                singleBits = sign | ((113 - shift) << 23) | ((normalized & 0x3ff) << 13);
            }
        } else if (exponent == 31) {
            singleBits = sign | 0x7f800000 | (mantissa << 13);
        } else {
            singleBits = sign | ((exponent + 112) << 23) | (mantissa << 13);
        }
        return Float.intBitsToFloat(singleBits);
    }

    public static final class Tensor {
        private final int[] shape;
        private final float[] data;

        private Tensor(int[] shape, float[] data) {
            this.shape = shape.clone();
            this.data = Objects.requireNonNull(data, "data");
        }

        public int rank() { return shape.length; }
        public int dimension(int axis) { return shape[axis]; }
        public int length() { return data.length; }
        public float[] values() { return data; }

        public void requireShape(int... expected) {
            if (!Arrays.equals(shape, expected)) {
                throw new IllegalArgumentException("Wrong Whisper tensor shape: expected "
                        + Arrays.toString(expected) + " but found " + Arrays.toString(shape));
            }
        }
    }
}
