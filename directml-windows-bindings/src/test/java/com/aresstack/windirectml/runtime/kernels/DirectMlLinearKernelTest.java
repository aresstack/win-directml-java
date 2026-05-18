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
 * End-to-End-Test für {@link DirectMlLinearKernel} – das erste echte
 * DirectML-Kernel-Dispatch des Projekts.
 * <p>
 * Validiert {@code y = x · W^T + b} gegen eine naive CPU-Referenz auf
 * gleichem (deterministischem) Float-Input. Toleranz ist großzügig, weil
 * der GPU-Kernel intern in einer anderen Reihenfolge summiert und das
 * letzte Mantissen-Bit bei großen K driftet.
 * <p>
 * Skippt sauber, wenn das D3D12/DML-Stack auf der Maschine nicht
 * verfügbar ist – analog zum bestehenden Roundtrip-Test.
 */
class DirectMlLinearKernelTest {

    private static final int M = 3;
    private static final int K = 8;
    private static final int N = 5;

    @Test
    void gemmWithBiasMatchesCpuReference() throws DirectMlRuntimeException {
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
            float[] xData = randomFloats(rng, M * K);
            float[] wData = randomFloats(rng, N * K);
            float[] bData = randomFloats(rng, N);

            float[] expected = cpuLinear(xData, wData, bData, M, K, N);

            CpuTensor xCpu = CpuTensor.float32(TensorShape.of(M, K), xData);
            CpuTensor wCpu = CpuTensor.float32(TensorShape.of(N, K), wData);
            CpuTensor bCpu = CpuTensor.float32(TensorShape.of(N), bData);

            try (GpuBuffer xBuf = ctx.allocateBufferFor(xCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer wBuf = ctx.allocateBufferFor(wCpu, GpuBuffer.BufferUsage.WEIGHT);
                 GpuBuffer bBuf = ctx.allocateBufferFor(bCpu, GpuBuffer.BufferUsage.WEIGHT);
                 GpuBuffer yBuf = ctx.allocateBuffer((long) M * N * Float.BYTES,
                         GpuBuffer.BufferUsage.ACTIVATION);
                 DirectMlLinearKernel kernel = new DirectMlLinearKernel(ctx, M, K, N, true)) {

                xBuf.upload(xCpu);
                wBuf.upload(wCpu);
                bBuf.upload(bCpu);

                DirectMlTensor x = new DirectMlTensor(TensorShape.of(M, K),
                        TensorLayout.rowMajor(TensorShape.of(M, K)),
                        TensorDataType.FLOAT32, xBuf);
                DirectMlTensor w = new DirectMlTensor(TensorShape.of(N, K),
                        TensorLayout.rowMajor(TensorShape.of(N, K)),
                        TensorDataType.FLOAT32, wBuf);
                DirectMlTensor b = new DirectMlTensor(TensorShape.of(N),
                        TensorLayout.rowMajor(TensorShape.of(N)),
                        TensorDataType.FLOAT32, bBuf);
                DirectMlTensor y = new DirectMlTensor(TensorShape.of(M, N),
                        TensorLayout.rowMajor(TensorShape.of(M, N)),
                        TensorDataType.FLOAT32, yBuf);

                kernel.dispatch(x, w, b, y);

                CpuTensor yOut = emptyCpu(M * N);
                yBuf.download(yOut);
                float[] actual = readFloats(yOut, M * N);

                assertArrayEquals(expected, actual, 1e-4f,
                        "DirectML GEMM must match the CPU reference for y = x · Wᵀ + b");
            }
        } finally {
            ctx.close();
        }
    }

    @Test
    void gemmWithoutBiasMatchesCpuReference() throws DirectMlRuntimeException {
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

            Random rng = new Random(0xDECAF);
            float[] xData = randomFloats(rng, M * K);
            float[] wData = randomFloats(rng, N * K);
            float[] expected = cpuLinear(xData, wData, null, M, K, N);

            CpuTensor xCpu = CpuTensor.float32(TensorShape.of(M, K), xData);
            CpuTensor wCpu = CpuTensor.float32(TensorShape.of(N, K), wData);

            try (GpuBuffer xBuf = ctx.allocateBufferFor(xCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer wBuf = ctx.allocateBufferFor(wCpu, GpuBuffer.BufferUsage.WEIGHT);
                 GpuBuffer yBuf = ctx.allocateBuffer((long) M * N * Float.BYTES,
                         GpuBuffer.BufferUsage.ACTIVATION);
                 DirectMlLinearKernel kernel = new DirectMlLinearKernel(ctx, M, K, N, false)) {

                xBuf.upload(xCpu);
                wBuf.upload(wCpu);

                DirectMlTensor x = new DirectMlTensor(TensorShape.of(M, K),
                        TensorLayout.rowMajor(TensorShape.of(M, K)),
                        TensorDataType.FLOAT32, xBuf);
                DirectMlTensor w = new DirectMlTensor(TensorShape.of(N, K),
                        TensorLayout.rowMajor(TensorShape.of(N, K)),
                        TensorDataType.FLOAT32, wBuf);
                DirectMlTensor y = new DirectMlTensor(TensorShape.of(M, N),
                        TensorLayout.rowMajor(TensorShape.of(M, N)),
                        TensorDataType.FLOAT32, yBuf);

                kernel.dispatch(x, w, null, y);

                CpuTensor yOut = emptyCpu(M * N);
                yBuf.download(yOut);
                float[] actual = readFloats(yOut, M * N);

                assertArrayEquals(expected, actual, 1e-4f,
                        "DirectML GEMM without bias must match the CPU reference");
            }
        } finally {
            ctx.close();
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────

    private static float[] randomFloats(Random rng, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (rng.nextFloat() - 0.5f) * 2.0f;
        }
        return out;
    }

    /**
     * y[m,n] = Σ_k x[m,k] * W[n,k]  (+ b[n] optional).
     */
    private static float[] cpuLinear(float[] x, float[] w, float[] b, int M, int K, int N) {
        float[] y = new float[M * N];
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float acc = 0f;
                for (int k = 0; k < K; k++) {
                    acc += x[m * K + k] * w[n * K + k];
                }
                if (b != null) acc += b[n];
                y[m * N + n] = acc;
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

