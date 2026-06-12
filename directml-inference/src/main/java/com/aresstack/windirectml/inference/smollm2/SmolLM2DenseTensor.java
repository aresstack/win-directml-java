package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyReferenceDenseOps;
import com.aresstack.windirectml.inference.model.SourceTensorDataType;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * Float32 view over a SmolLM2 runtime tensor payload.
 *
 * <p><b>Heap-light FP32 path (slice H2c):</b> when the payload is FLOAT32, the float decode is <em>deferred</em>:
 * the raw little-endian mmap {@link ByteBuffer} slice is retained via {@link #fp32LittleEndianSource()} so the WARP
 * dense projections can upload it directly (no host {@code float[]}). The {@code float[]} is only materialised on first
 * access ({@link #copyValues()}, {@link #value(int)}, {@link #multiplyVector(float[])}, {@link #dotRow(int, float[])},
 * row copies) — i.e. for reference/diagnostic/CPU paths and fused projections. Validation (shape, payload length) stays
 * eager. FLOAT16/BFLOAT16 tensors keep an eager {@code float[]} and expose no FP32 source. This also removes the former
 * double copy (decode {@code float[]} + ctor {@code clone()}).</p>
 */
public final class SmolLM2DenseTensor {

    private final String name;
    private final long[] dims;
    private final int elementCount;
    private final ByteBuffer fp32SourceLe;   // nullable: present only for FLOAT32 mmap-backed tensors
    private volatile float[] values;         // lazily decoded for fp32SourceLe; eager for FP16/BF16

    private SmolLM2DenseTensor(String name, long[] dims, int elementCount, ByteBuffer fp32SourceLe, float[] values) {
        this.name = Objects.requireNonNull(name, "name");
        this.dims = dims.clone();
        this.elementCount = elementCount;
        this.fp32SourceLe = fp32SourceLe;
        this.values = values;
    }

    public static SmolLM2DenseTensor from(SmolLM2WeightTensor tensor) {
        Objects.requireNonNull(tensor, "tensor");
        long[] dims = tensor.dims();
        int elementCount = checkedElementCount(dims);
        SourceTensorDataType dataType = SourceTensorDataType.fromOnnxCode(tensor.dataType());
        ByteBuffer raw = tensor.rawDataBuffer().order(ByteOrder.LITTLE_ENDIAN);
        int expectedBytes = Math.multiplyExact(elementCount, dataType.bytesPerElement());
        if (raw.remaining() < expectedBytes) {
            throw new IllegalStateException("Tensor payload is shorter than its shape requires: "
                    + tensor.tensorName());
        }
        if ("FLOAT".equals(dataType.name())) {
            // Heap-light: keep the LE slice, decode lazily only if a float[] is actually requested.
            return new SmolLM2DenseTensor(tensor.tensorName(), dims, elementCount, raw, null);
        }
        // FLOAT16 / BFLOAT16: eager conversion to FP32 (no direct FP32 source buffer for upload).
        float[] decoded = decode(raw, dataType, elementCount, tensor.tensorName());
        return new SmolLM2DenseTensor(tensor.tensorName(), dims, elementCount, null, decoded);
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
        return elementCount;
    }

    public float value(int index) {
        return ensureValues()[index];
    }

    public float[] copyValues() {
        float[] decoded = ensureValues();
        return Arrays.copyOf(decoded, decoded.length);
    }

    public float[] copyRow(int rowIndex) {
        requireRank(2);
        int rows = dim(0);
        int cols = dim(1);
        if (rowIndex < 0 || rowIndex >= rows) {
            throw new IllegalArgumentException("rowIndex out of range for " + name + ": " + rowIndex);
        }
        float[] decoded = ensureValues();
        return Arrays.copyOfRange(decoded, rowIndex * cols, rowIndex * cols + cols);
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
        System.arraycopy(ensureValues(), rowIndex * cols, target, 0, cols);
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
        DecoderOnlyReferenceDenseOps.multiplyRows(ensureValues(), rows, cols, input, output);
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
        return DecoderOnlyReferenceDenseOps.dot(ensureValues(), rowIndex * cols, input, 0, cols);
    }

    /**
     * The raw little-endian FP32 payload slice for direct (heap-light) WARP upload, or {@code null} when this tensor
     * is not FLOAT32 (FLOAT16/BFLOAT16). Each call returns an independent view (own position/limit) over the shared
     * mmap region; the caller must not rely on the mapping outliving the runtime tensor catalog.
     */
    ByteBuffer fp32LittleEndianSource() {
        return fp32SourceLe == null ? null : fp32SourceLe.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    }

    private float[] ensureValues() {
        float[] v = values;
        if (v == null) {
            // Idempotent lazy FP32 decode (load-time / single-threaded use); publication via the volatile field.
            ByteBuffer src = fp32SourceLe.duplicate().order(ByteOrder.LITTLE_ENDIAN);
            float[] decoded = new float[elementCount];
            for (int i = 0; i < elementCount; i++) {
                decoded[i] = src.getFloat();
            }
            v = decoded;
            values = v;
        }
        return v;
    }

    private void requireRank(int expectedRank) {
        if (dims.length != expectedRank) {
            throw new IllegalStateException(name + " must have rank " + expectedRank
                    + " but has " + dims.length);
        }
    }

    private static float[] decode(ByteBuffer raw, SourceTensorDataType dataType, int elementCount, String tensorName) {
        float[] values = new float[elementCount];
        for (int i = 0; i < elementCount; i++) {
            values[i] = switch (dataType.name()) {
                case "FLOAT" -> raw.getFloat();
                case "FLOAT16" -> halfToFloat(raw.getShort());
                case "BFLOAT16" -> Float.intBitsToFloat((raw.getShort() & 0xFFFF) << 16);
                default -> throw new IllegalStateException("Unsupported SmolLM2 runtime dtype "
                        + dataType.name() + " for tensor " + tensorName);
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
