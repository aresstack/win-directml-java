package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorEntry;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * A retained <b>BF16</b> host view of a row-major {@code [rows, cols]} weight tensor (GEMMA-BF16-PACK-2).
 *
 * <p>For the tied embedding / LM head the SafeTensors payload is already BF16, but the heap-light loader
 * widened it to a retained FP32 {@link ByteBuffer} ({@code rows*cols*4} bytes) so the per-token embedding
 * lookup and the one-off LM-head upload could read floats. This view keeps the weights as their original
 * BF16 bytes ({@code rows*cols*2}, ~half the RAM) and decodes to FP32 only where needed: one row at a time
 * for the embedding lookup, or a single transient full widen for the LM-head device upload (the device
 * buffer stays FP32 — the DML GEMM is FP32). Lossless: BF16→FP32 is the exact widen
 * ({@code float = bf16_bits << 16}), identical to {@link Gemma3WeightBufferView#decodeFp32LittleEndian} for
 * a BF16 source. No {@code .wdmlpack} format change — same payload, a smaller retained decode target.</p>
 */
public final class Gemma3Bf16WeightView {

    private final ByteBuffer bf16;   // direct, little-endian, rows*cols BF16 shorts (position 0)
    private final int rows;
    private final int cols;

    private Gemma3Bf16WeightView(ByteBuffer bf16, int rows, int cols) {
        this.bf16 = bf16;
        this.rows = rows;
        this.cols = cols;
    }

    /**
     * Copy a {@code [rows, cols]} BF16 SafeTensors tensor into a fresh direct little-endian BF16 buffer
     * (independent of the mmap'd payload, like the FP32 decode but half the size). The entry must be BF16.
     */
    public static Gemma3Bf16WeightView ofBf16Copy(SafeTensorEntry entry) {
        if (!"BF16".equals(entry.dtype())) {
            throw new IllegalArgumentException("Gemma3Bf16WeightView requires a BF16 tensor, got " + entry.dtype());
        }
        long[] shape = entry.shape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("expected a 2-D [rows, cols] tensor, got shape length " + shape.length);
        }
        int rows = Math.toIntExact(shape[0]);
        int cols = Math.toIntExact(shape[1]);
        ByteBuffer src = entry.dataBuffer().duplicate().order(ByteOrder.LITTLE_ENDIAN);
        long expected = (long) rows * cols * 2L;
        if (src.remaining() != expected) {
            throw new IllegalArgumentException("BF16 byte size mismatch for [" + rows + ", " + cols
                    + "]: remaining=" + src.remaining() + ", expected=" + expected);
        }
        ByteBuffer copy = ByteBuffer.allocateDirect(Math.toIntExact(expected)).order(ByteOrder.LITTLE_ENDIAN);
        copy.put(src);
        copy.flip();
        return new Gemma3Bf16WeightView(copy, rows, cols);
    }

    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }

    /** Retained host bytes of this view (BF16: rows*cols*2). */
    public long retainedBytes() {
        return (long) rows * cols * 2L;
    }

    /**
     * Decode row {@code r} ({@code cols} values), scaled, into {@code dst[0..cols)} — BF16→FP32 widen then
     * multiply by {@code scale}. Used by the embedding lookup (only the prompt's token rows are decoded).
     */
    public void decodeRowScaled(int r, float scale, float[] dst) {
        if (r < 0 || r >= rows) {
            throw new IllegalArgumentException("row out of range: " + r + " (rows=" + rows + ")");
        }
        if (dst.length < cols) {
            throw new IllegalArgumentException("dst too short: " + dst.length + " < cols=" + cols);
        }
        int base = r * cols * 2;
        for (int i = 0; i < cols; i++) {
            dst[i] = bf16ToFloat(bf16.getShort(base + i * 2)) * scale;
        }
    }

    /**
     * One transient full widen to a fresh direct little-endian FP32 {@link ByteBuffer} (for the one-off
     * LM-head device upload). The caller uploads it and lets it be collected — it is not retained.
     */
    public ByteBuffer inflateToFp32() {
        int n = rows * cols;
        ByteBuffer out = ByteBuffer.allocateDirect(Math.multiplyExact(n, Float.BYTES)).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            out.putFloat(bf16ToFloat(bf16.getShort(i * 2)));
        }
        out.flip();
        return out;
    }

    /** BF16 → FP32: BF16 is the high 16 bits of the IEEE-754 float. */
    private static float bf16ToFloat(short bits) {
        return Float.intBitsToFloat((bits & 0xFFFF) << 16);
    }
}
