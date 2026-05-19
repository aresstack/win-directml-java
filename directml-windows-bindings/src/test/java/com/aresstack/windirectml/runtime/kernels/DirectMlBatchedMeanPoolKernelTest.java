package com.aresstack.windirectml.runtime.kernels;

import com.aresstack.windirectml.runtime.CpuTensor;
import com.aresstack.windirectml.runtime.DirectMlContextImpl;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;
import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.GpuBuffer;
import com.aresstack.windirectml.runtime.TensorDataType;
import com.aresstack.windirectml.runtime.TensorLayout;
import com.aresstack.windirectml.runtime.TensorShape;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GPU-Parity-Test für {@link DirectMlBatchedMeanPoolKernel}: vergleicht
 * {@code y[n,h] = Σ_t w[n,t] · x[n,t,h]} (single batched GEMM dispatch)
 * gegen eine naive CPU-Referenz auf deterministischen Float-Daten.
 * <p>
 * Geprüft wird mit N=3 Zeilen, einer per-row variierenden „validCount"
 * Maske und einer nicht-2-Potenz-Sequenzlänge, damit DirectML's
 * Pad-/Stride-Behandlung mit getroffen wird.
 */
class DirectMlBatchedMeanPoolKernelTest {

    private static final int N = 3;
    private static final int S = 17;
    private static final int H = 32;

    @Test
    void batchedMeanPoolMatchesCpuReference() throws DirectMlRuntimeException {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12");

        DirectMlContextImpl ctx = new DirectMlContextImpl("directml");
        try {
            try {
                ctx.initialize();
            } catch (DirectMlRuntimeException e) {
                assumeTrue(false, "D3D12/DML stack not available: " + e.getMessage());
                return;
            }
            assumeTrue(ctx.isReady() && ctx.bindings().hasDirectMl(),
                    "Skipping: no DirectML device on this adapter");

            Random rng = new Random(0xC0FFEE);
            float[] xData = randomFloats(rng, N * S * H);

            // Per-row varying valid count → exercises per-row weight rows
            // independently.
            int[] valid = new int[]{12, 7, 17};
            float[] wData = new float[N * S];
            for (int n = 0; n < N; n++) {
                float inv = 1.0f / valid[n];
                int base = n * S;
                for (int t = 0; t < valid[n]; t++) wData[base + t] = inv;
            }

            float[] expected = cpuBatchedMeanPool(xData, wData, N, S, H);

            CpuTensor xCpu = CpuTensor.float32(TensorShape.of(N, S, H), xData);
            CpuTensor wCpu = CpuTensor.float32(TensorShape.of(N, S), wData);

            try (GpuBuffer xBuf = ctx.allocateBufferFor(xCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer wBuf = ctx.allocateBufferFor(wCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer yBuf = ctx.allocateBuffer((long) N * H * Float.BYTES,
                         GpuBuffer.BufferUsage.ACTIVATION);
                 DirectMlBatchedMeanPoolKernel kernel =
                         new DirectMlBatchedMeanPoolKernel(ctx, N, S, H)) {

                xBuf.upload(xCpu);
                wBuf.upload(wCpu);

                DirectMlTensor x = new DirectMlTensor(TensorShape.of(N, S, H),
                        TensorLayout.rowMajor(TensorShape.of(N, S, H)),
                        TensorDataType.FLOAT32, xBuf);
                DirectMlTensor w = new DirectMlTensor(TensorShape.of(N, S),
                        TensorLayout.rowMajor(TensorShape.of(N, S)),
                        TensorDataType.FLOAT32, wBuf);
                DirectMlTensor y = new DirectMlTensor(TensorShape.of(N, H),
                        TensorLayout.rowMajor(TensorShape.of(N, H)),
                        TensorDataType.FLOAT32, yBuf);

                kernel.dispatch(x, w, y);

                CpuTensor yOut = emptyCpu(N * H);
                yBuf.download(yOut);
                float[] actual = readFloats(yOut, N * H);

                assertArrayEquals(expected, actual, 1e-4f,
                        "Batched DirectML mean-pool must match the CPU reference for all N rows");
            }
        } finally {
            ctx.close();
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────

    private static float[] randomFloats(Random rng, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) out[i] = (rng.nextFloat() - 0.5f) * 2.0f;
        return out;
    }

    private static float[] cpuBatchedMeanPool(float[] x, float[] w, int N, int S, int H) {
        float[] y = new float[N * H];
        for (int n = 0; n < N; n++) {
            int wBase = n * S;
            int xBase = n * S * H;
            int yBase = n * H;
            for (int t = 0; t < S; t++) {
                float wt = w[wBase + t];
                if (wt == 0f) continue;
                int row = xBase + t * H;
                for (int h = 0; h < H; h++) y[yBase + h] += wt * x[row + h];
            }
        }
        return y;
    }

    private static CpuTensor emptyCpu(int n) {
        ByteBuffer storage = ByteBuffer.allocateDirect(n * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        TensorShape shape = TensorShape.of(n);
        return new CpuTensor(shape, TensorLayout.rowMajor(shape), TensorDataType.FLOAT32, storage);
    }

    private static float[] readFloats(CpuTensor t, int n) {
        float[] out = new float[n];
        FloatBuffer fv = t.data().duplicate().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        fv.position(0);
        fv.get(out, 0, n);
        return out;
    }
}

