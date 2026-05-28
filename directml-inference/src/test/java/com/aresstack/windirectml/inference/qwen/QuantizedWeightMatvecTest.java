package com.aresstack.windirectml.inference.qwen;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Correctness tests for {@link Qwen2Weights.QuantizedWeight#matvec} and
 * {@link Qwen2Weights.QuantizedWeight#matmul} after the Prio-2 SIMD optimisation
 * (thread-local block buffer + {@link SimdOps#dot}).
 *
 * <p>Each test compares the SIMD path output against an independently-computed
 * reference to guard against nibble-extraction or sign errors.
 */
class QuantizedWeightMatvecTest {

    private static final int BLOCK_SIZE = 128;

    // ── helpers ──────────────────────────────────────────────────────────────

    /**
     * Build a QuantizedWeight where every weight nibble = {@code nibbleVal},
     * every zero-point = {@code zpNibble}, every scale = {@code scale},
     * for {@code N} output rows and {@code K = BLOCK_SIZE} input features.
     *
     * <p>Weight packing (low nibble first per byte, row-major):
     * byte[b] = (nibble_even & 0xF) | ((nibble_odd & 0xF) << 4)
     */
    private static Qwen2Weights.QuantizedWeight uniformWeight(int N, int nibbleVal,
                                                              int zpNibble, float scale) {
        int K = BLOCK_SIZE;
        int blocksPerRow = K / BLOCK_SIZE;  // = 1
        int qWeightLen = N * blocksPerRow * (BLOCK_SIZE / 2);
        byte[] qw = new byte[qWeightLen];
        byte packed = (byte) ((nibbleVal & 0xF) | ((nibbleVal & 0xF) << 4));
        Arrays.fill(qw, packed);

        float[] scales = new float[N * blocksPerRow];
        Arrays.fill(scales, scale);

        // Zero-points: 2 per byte, low nibble first
        int numZpBytes = (N * blocksPerRow + 1) / 2;
        byte[] zp = new byte[numZpBytes];
        byte zpPacked = (byte) ((zpNibble & 0xF) | ((zpNibble & 0xF) << 4));
        Arrays.fill(zp, zpPacked);

        return new Qwen2Weights.QuantizedWeight(qw, scales, zp, N, K, BLOCK_SIZE);
    }

    /**
     * Scalar reference implementation (independent of the optimised code).
     */
    private static void matvecRef(Qwen2Weights.QuantizedWeight qw, float[] x, float[] y) {
        int N = qw.N();
        int K = qw.K();
        int bs = qw.blockSize();
        int bpr = K / bs;
        byte[] weights = qw.qWeight();
        float[] sc = qw.scales();
        byte[] zpArr = qw.zeroPoints();
        for (int n = 0; n < N; n++) {
            float sum = 0f;
            int qOff = n * bpr * (bs / 2);
            int scOff = n * bpr;
            for (int blk = 0; blk < bpr; blk++) {
                float scale = sc[scOff + blk];
                int zpIdx = n * bpr + blk;
                int zpByte = zpArr[zpIdx / 2] & 0xFF;
                int zp = (zpIdx % 2 == 0) ? (zpByte & 0xF) : (zpByte >>> 4);
                int kBase = blk * bs;
                int qBase = qOff + blk * (bs / 2);
                for (int j = 0; j < bs / 2; j++) {
                    int b = weights[qBase + j] & 0xFF;
                    sum += x[kBase + 2 * j] * ((b & 0xF) - zp) * scale;
                    sum += x[kBase + 2 * j + 1] * ((b >>> 4) - zp) * scale;
                }
            }
            y[n] += sum;
        }
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    @Test
    void matvecAllOnesNibbleZpZero() {
        // All nibbles = 1, zp = 0, scale = 1.0 → each output row sums 128 × 1 × 1 = 128
        int N = 4;
        Qwen2Weights.QuantizedWeight qw = uniformWeight(N, /*nibble*/ 1, /*zp*/ 0, 1.0f);
        float[] x = new float[BLOCK_SIZE];
        Arrays.fill(x, 1.0f);
        float[] y = new float[N];   // zero-initialized
        qw.matvec(x, y);
        for (int n = 0; n < N; n++) {
            assertEquals(BLOCK_SIZE, y[n], 1e-3f,
                    "row " + n + ": expected 128.0 but got " + y[n]);
        }
    }

    @Test
    void matvecNibble8Zp8ScaleOneGivesZero() {
        // nibble = 8, zp = 8, scale = 1.0 → weight = (8 - 8) × 1 = 0
        int N = 4;
        Qwen2Weights.QuantizedWeight qw = uniformWeight(N, 8, 8, 1.0f);
        float[] x = new float[BLOCK_SIZE];
        Arrays.fill(x, 5.0f);
        float[] y = new float[N];
        qw.matvec(x, y);
        for (int n = 0; n < N; n++) {
            assertEquals(0.0f, y[n], 1e-6f, "row " + n + " should be 0.0");
        }
    }

    @Test
    void matvecScalePropagates() {
        // nibble = 2, zp = 0, scale = 0.5 → weight = 2 × 0.5 = 1.0
        // x = all 1.0, output should be 128 × 1 = 128
        int N = 2;
        Qwen2Weights.QuantizedWeight qw = uniformWeight(N, 2, 0, 0.5f);
        float[] x = new float[BLOCK_SIZE];
        Arrays.fill(x, 1.0f);
        float[] y = new float[N];
        qw.matvec(x, y);
        for (int n = 0; n < N; n++) {
            assertEquals(BLOCK_SIZE, y[n], 1e-3f,
                    "row " + n + ": scale test failed, got " + y[n]);
        }
    }

    @Test
    void matvecAccumulatesIntoExistingY() {
        // Verify that matvec does y[n] += sum (not y[n] = sum)
        int N = 2;
        Qwen2Weights.QuantizedWeight qw = uniformWeight(N, 1, 0, 1.0f);
        float[] x = new float[BLOCK_SIZE];
        Arrays.fill(x, 1.0f);
        float[] y = {10.0f, 20.0f};
        qw.matvec(x, y);
        assertEquals(10.0f + BLOCK_SIZE, y[0], 1e-3f);
        assertEquals(20.0f + BLOCK_SIZE, y[1], 1e-3f);
    }

    @Test
    void matvecMatchesReferenceForLargerN() {
        // Compare SIMD output vs. independent scalar reference for N=16
        int N = 16;
        Qwen2Weights.QuantizedWeight qw = uniformWeight(N, 3, 1, 2.0f);
        // x: non-trivial values
        float[] x = new float[BLOCK_SIZE];
        for (int i = 0; i < BLOCK_SIZE; i++) x[i] = (float) Math.sin(i * 0.1);

        float[] ySimd = new float[N];
        float[] yRef = new float[N];

        qw.matvec(x, ySimd);
        matvecRef(qw, x, yRef);

        for (int n = 0; n < N; n++) {
            assertEquals(yRef[n], ySimd[n], Math.abs(yRef[n]) * 1e-4f + 1e-4f,
                    "row " + n);
        }
    }

    @Test
    void matmulMatchesRepeatedMatvecCalls() {
        // matmul([x0; x1], y) should equal two separate matvec calls
        int N = 8;
        Qwen2Weights.QuantizedWeight qw = uniformWeight(N, 5, 2, 0.25f);

        float[] x0 = new float[BLOCK_SIZE];
        float[] x1 = new float[BLOCK_SIZE];
        for (int i = 0; i < BLOCK_SIZE; i++) {
            x0[i] = (float) Math.cos(i * 0.05);
            x1[i] = (float) Math.sin(i * 0.07 + 1.0);
        }

        // matmul path
        float[] xBatch = new float[2 * BLOCK_SIZE];
        System.arraycopy(x0, 0, xBatch, 0, BLOCK_SIZE);
        System.arraycopy(x1, 0, xBatch, BLOCK_SIZE, BLOCK_SIZE);
        float[] yBatch = new float[2 * N];
        qw.matmul(xBatch, yBatch, 2);

        // Reference: two individual matvec calls
        float[] yRef0 = new float[N];
        float[] yRef1 = new float[N];
        qw.matvec(x0, yRef0);
        qw.matvec(x1, yRef1);

        for (int n = 0; n < N; n++) {
            assertEquals(yRef0[n], yBatch[n], Math.abs(yRef0[n]) * 1e-4f + 1e-4f, "batch[0] row " + n);
            assertEquals(yRef1[n], yBatch[N + n], Math.abs(yRef1[n]) * 1e-4f + 1e-4f, "batch[1] row " + n);
        }
    }
}
