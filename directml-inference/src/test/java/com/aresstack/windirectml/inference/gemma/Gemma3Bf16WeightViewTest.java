package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorEntry;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * GEMMA-BF16-PACK-2: the retained BF16 weight view ({@link Gemma3Bf16WeightView}). Row/full decode must be
 * the exact BF16→FP32 widen ({@code float = bf16_bits << 16}) — what the embedding lookup and the LM-head
 * upload rely on — plus bounds/shape/dtype guards. Pure CPU, runs everywhere.
 */
class Gemma3Bf16WeightViewTest {

    @Test
    void decodeRowAndInflateMatchTheBf16Widen() {
        int rows = 9;
        int cols = 7;
        Random rng = new Random(42);
        short[] bits = new short[rows * cols];
        for (int i = 0; i < bits.length; i++) {
            bits[i] = (short) rng.nextInt(0x10000);
        }
        Gemma3Bf16WeightView view = viewOf(rows, cols, bits);

        float scale = 1.37f;
        for (int r = 0; r < rows; r++) {
            float[] dst = new float[cols];
            view.decodeRowScaled(r, scale, dst);
            float[] want = new float[cols];
            for (int c = 0; c < cols; c++) {
                want[c] = widen(bits[r * cols + c]) * scale;
            }
            assertArrayEquals(want, dst, 0f, "row " + r);
        }

        ByteBuffer fp32 = view.inflateToFp32();
        assertEquals((long) rows * cols * Float.BYTES, fp32.limit());
        for (int i = 0; i < rows * cols; i++) {
            assertEquals(widen(bits[i]), fp32.getFloat(i * Float.BYTES), 0f, "elem " + i);
        }
        assertEquals((long) rows * cols * 2L, view.retainedBytes());
        assertEquals(rows, view.rows());
        assertEquals(cols, view.cols());
    }

    @Test
    void boundsAndShapeGuards() {
        Gemma3Bf16WeightView view = viewOf(4, 3, new short[12]);
        assertThrows(IllegalArgumentException.class, () -> view.decodeRowScaled(-1, 1f, new float[3]));
        assertThrows(IllegalArgumentException.class, () -> view.decodeRowScaled(4, 1f, new float[3]));
        assertThrows(IllegalArgumentException.class, () -> view.decodeRowScaled(0, 1f, new float[2]));
    }

    @Test
    void rejectsNonBf16AndNon2dTensors() {
        // Non-BF16 dtype.
        assertThrows(IllegalArgumentException.class, () -> Gemma3Bf16WeightView.ofBf16Copy(
                new SafeTensorEntry("e", "F32", 0, new long[]{2, 2}, 0, 16, 16,
                        ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN))));
        // Not 2-D.
        assertThrows(IllegalArgumentException.class, () -> Gemma3Bf16WeightView.ofBf16Copy(
                new SafeTensorEntry("e", "BF16", 0, new long[]{8}, 0, 16, 16,
                        ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN))));
    }

    private static Gemma3Bf16WeightView viewOf(int rows, int cols, short[] bits) {
        ByteBuffer buf = ByteBuffer.allocate(rows * cols * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (short s : bits) {
            buf.putShort(s);
        }
        buf.flip();
        SafeTensorEntry entry = new SafeTensorEntry("emb", "BF16", 0, new long[]{rows, cols},
                0, buf.limit(), buf.limit(), buf);
        return Gemma3Bf16WeightView.ofBf16Copy(entry);
    }

    private static float widen(short bf16) {
        return Float.intBitsToFloat((bf16 & 0xFFFF) << 16);
    }
}
