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
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GPU-Parity-Test für {@link DirectMlBatchedL2NormalizeKernel}.
 * <p>
 * Prüft, dass alle {@code N} Zeilen unabhängig per L2-Norm normalisiert
 * werden (jede Zeile mit eigenem Skalar), und dass die Ausgaben Bit-nah
 * an der CPU-Referenz {@code y = x / max(‖x‖, ε)} liegen.
 */
class DirectMlBatchedL2NormalizeKernelTest {

    private static final int N = 4;
    private static final int H = 384;
    private static final float EPSILON = 1e-12f;

    @Test
    void batchedL2NormalizeMatchesCpuReference() throws DirectMlRuntimeException {
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

            Random rng = new Random(0xFEED);
            float[] xData = new float[N * H];
            // Vary the magnitude per row so each row needs its own norm.
            for (int n = 0; n < N; n++) {
                float scale = 0.1f + n * 0.7f;
                for (int h = 0; h < H; h++) {
                    xData[n * H + h] = scale * (rng.nextFloat() - 0.5f) * 2.0f;
                }
            }
            float[] expected = cpuBatchedL2Normalize(xData, N, H, EPSILON);

            CpuTensor xCpu = CpuTensor.float32(TensorShape.of(N, H), xData);

            try (GpuBuffer xBuf = ctx.allocateBufferFor(xCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer yBuf = ctx.allocateBuffer((long) N * H * Float.BYTES,
                         GpuBuffer.BufferUsage.ACTIVATION);
                 DirectMlBatchedL2NormalizeKernel kernel =
                         new DirectMlBatchedL2NormalizeKernel(ctx, N, H, EPSILON)) {

                xBuf.upload(xCpu);

                DirectMlTensor x = new DirectMlTensor(TensorShape.of(N, H),
                        TensorLayout.rowMajor(TensorShape.of(N, H)),
                        TensorDataType.FLOAT32, xBuf);
                DirectMlTensor y = new DirectMlTensor(TensorShape.of(N, H),
                        TensorLayout.rowMajor(TensorShape.of(N, H)),
                        TensorDataType.FLOAT32, yBuf);

                kernel.dispatch(x, y, EPSILON);

                CpuTensor yOut = emptyCpu(N * H);
                yBuf.download(yOut);
                float[] actual = readFloats(yOut, N * H);

                assertArrayEquals(expected, actual, 1e-5f,
                        "Batched DirectML L2-normalize must match CPU per-row reference");

                // Every row must have unit L2 norm.
                for (int n = 0; n < N; n++) {
                    double norm = 0.0;
                    for (int h = 0; h < H; h++) {
                        float v = actual[n * H + h];
                        norm += (double) v * v;
                    }
                    assertTrue(Math.abs(Math.sqrt(norm) - 1.0) < 1e-5,
                            "Row " + n + " must be unit-norm, got " + Math.sqrt(norm));
                }

                assertEquals(N, kernel.batch());
                assertEquals(H, kernel.hidden());
                assertEquals(EPSILON, kernel.epsilon());
            }
        } finally {
            ctx.close();
        }
    }

    @Test
    void dispatchRejectsMismatchedEpsilon() throws DirectMlRuntimeException {
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

            float[] xData = new float[N * H];
            xData[0] = 1.0f;
            CpuTensor xCpu = CpuTensor.float32(TensorShape.of(N, H), xData);

            try (GpuBuffer xBuf = ctx.allocateBufferFor(xCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer yBuf = ctx.allocateBuffer((long) N * H * Float.BYTES,
                         GpuBuffer.BufferUsage.ACTIVATION);
                 DirectMlBatchedL2NormalizeKernel kernel =
                         new DirectMlBatchedL2NormalizeKernel(ctx, N, H, EPSILON)) {

                xBuf.upload(xCpu);
                DirectMlTensor x = new DirectMlTensor(TensorShape.of(N, H),
                        TensorLayout.rowMajor(TensorShape.of(N, H)), TensorDataType.FLOAT32, xBuf);
                DirectMlTensor y = new DirectMlTensor(TensorShape.of(N, H),
                        TensorLayout.rowMajor(TensorShape.of(N, H)), TensorDataType.FLOAT32, yBuf);

                assertThrows(DirectMlRuntimeException.class,
                        () -> kernel.dispatch(x, y, 1e-6f));
            }
        } finally {
            ctx.close();
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────

    private static float[] cpuBatchedL2Normalize(float[] x, int N, int H, float epsilon) {
        float[] y = new float[N * H];
        for (int n = 0; n < N; n++) {
            int base = n * H;
            double sum = 0.0;
            for (int h = 0; h < H; h++) sum += (double) x[base + h] * x[base + h];
            double norm = Math.sqrt(sum);
            double inv = 1.0 / Math.max(norm, epsilon);
            for (int h = 0; h < H; h++) y[base + h] = (float) (x[base + h] * inv);
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

