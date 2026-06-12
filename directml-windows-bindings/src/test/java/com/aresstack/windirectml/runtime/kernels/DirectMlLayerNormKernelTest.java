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
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Validiert {@link DirectMlLayerNormKernel} (BERT-Style LayerNorm) gegen
 * eine CPU-Referenz.
 */
class DirectMlLayerNormKernelTest {

    private static final int M = 4;
    private static final int H = 16;
    private static final float EPS = 1e-12f;

    @Test
    void layerNormMatchesCpuReference() throws DirectMlRuntimeException {
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

            Random rng = new Random(0xBA5E);
            float[] xData = randomFloats(rng, M * H);
            float[] gammaData = randomFloats(rng, H);
            float[] betaData = randomFloats(rng, H);

            float[] expected = cpuLayerNorm(xData, gammaData, betaData, M, H, EPS);

            CpuTensor xCpu = CpuTensor.float32(TensorShape.of(M, H), xData);
            CpuTensor gCpu = CpuTensor.float32(TensorShape.of(H), gammaData);
            CpuTensor bCpu = CpuTensor.float32(TensorShape.of(H), betaData);

            try (GpuBuffer xBuf = ctx.allocateBufferFor(xCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer gBuf = ctx.allocateBufferFor(gCpu, GpuBuffer.BufferUsage.WEIGHT);
                 GpuBuffer bBuf = ctx.allocateBufferFor(bCpu, GpuBuffer.BufferUsage.WEIGHT);
                 GpuBuffer yBuf = ctx.allocateBuffer((long) M * H * Float.BYTES,
                         GpuBuffer.BufferUsage.ACTIVATION);
                 DirectMlLayerNormKernel kernel = new DirectMlLayerNormKernel(ctx, M, H, EPS)) {

                xBuf.upload(xCpu);
                gBuf.upload(gCpu);
                bBuf.upload(bCpu);

                DirectMlTensor x = tensor2D(xBuf, M, H);
                DirectMlTensor g = tensor1D(gBuf, H);
                DirectMlTensor b = tensor1D(bBuf, H);
                DirectMlTensor y = tensor2D(yBuf, M, H);

                kernel.dispatch(x, g, b, y, EPS);

                CpuTensor yOut = emptyCpu(M * H);
                yBuf.download(yOut);
                float[] actual = readFloats(yOut, M * H);

                assertArrayEquals(expected, actual, 1e-4f,
                        "DirectML LayerNorm must match the CPU reference");
            }
        } finally {
            ctx.close();
        }
    }

    /**
     * Regression guard for the {@code bucket=64} WARP device-removal that broke
     * MiniLM embeddings and the cross-encoder reranker.
     * <p>
     * Root cause: a row-wise kernel whose operator reports {@code tempSize == 0}
     * (LayerNorm via MVN0) still bound the context's shared temporary buffer as
     * a {@code DML_BINDING_TYPE_BUFFER} of size 0 once <em>another</em> kernel
     * (a Linear with {@code tempSize > 0}) had caused that buffer to be
     * allocated. Binding a temporary to an operator that needs none removed the
     * device. The bug was invisible at the tiny synthetic dims of the other
     * tests because their GEMMs report {@code tempSize == 0}, so the shared
     * buffer never existed.
     * <p>
     * This test forces the shared temp buffer to exist (via a Linear with a
     * non-zero {@code tempSize}) and then runs a {@code tempSize == 0} LayerNorm
     * on the same context, asserting it still produces the correct result.
     */
    @Test
    void layerNormRunsWhenSharedTempBufferExists() throws DirectMlRuntimeException {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12");

        final int lnM = 64, lnH = 384;   // real BERT-ish dims (MiniLM embedding LayerNorm)
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

            // Force allocation of the context's shared temp buffer via a GEMM
            // whose operator reports tempSize > 0 (mirrors a BERT MLP Linear).
            try (DirectMlLinearKernel forcer = new DirectMlLinearKernel(ctx, lnM, lnH, 1536, true)) {
                assumeTrue(ctx.getSharedTempSize() > 0,
                        "this adapter reports tempSize==0 for the GEMM; regression not reproducible here");

                Random rng = new Random(0x10E0);
                float[] xData = randomFloats(rng, lnM * lnH);
                float[] gammaData = randomFloats(rng, lnH);
                float[] betaData = randomFloats(rng, lnH);
                float[] expected = cpuLayerNorm(xData, gammaData, betaData, lnM, lnH, EPS);

                CpuTensor xCpu = CpuTensor.float32(TensorShape.of(lnM, lnH), xData);
                CpuTensor gCpu = CpuTensor.float32(TensorShape.of(lnH), gammaData);
                CpuTensor bCpu = CpuTensor.float32(TensorShape.of(lnH), betaData);

                try (GpuBuffer xBuf = ctx.allocateBufferFor(xCpu, GpuBuffer.BufferUsage.ACTIVATION);
                     GpuBuffer gBuf = ctx.allocateBufferFor(gCpu, GpuBuffer.BufferUsage.WEIGHT);
                     GpuBuffer bBuf = ctx.allocateBufferFor(bCpu, GpuBuffer.BufferUsage.WEIGHT);
                     GpuBuffer yBuf = ctx.allocateBuffer((long) lnM * lnH * Float.BYTES,
                             GpuBuffer.BufferUsage.ACTIVATION);
                     DirectMlLayerNormKernel kernel = new DirectMlLayerNormKernel(ctx, lnM, lnH, EPS)) {

                    xBuf.upload(xCpu);
                    gBuf.upload(gCpu);
                    bBuf.upload(bCpu);

                    // Pre-fix this dispatch removed the device (DXGI_ERROR_DEVICE_REMOVED
                    // surfaced at the next CreateBindingTable).
                    kernel.dispatch(tensor2D(xBuf, lnM, lnH), tensor1D(gBuf, lnH),
                            tensor1D(bBuf, lnH), tensor2D(yBuf, lnM, lnH), EPS);

                    CpuTensor yOut = emptyCpu(lnM * lnH);
                    yBuf.download(yOut);
                    float[] actual = readFloats(yOut, lnM * lnH);
                    assertArrayEquals(expected, actual, 1e-4f,
                            "LayerNorm (tempSize==0) must run correctly while the shared temp buffer exists");
                }
            }
        } finally {
            ctx.close();
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────

    private static DirectMlTensor tensor2D(GpuBuffer buf, int m, int h) {
        TensorShape s = TensorShape.of(m, h);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private static DirectMlTensor tensor1D(GpuBuffer buf, int n) {
        TensorShape s = TensorShape.of(n);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private static float[] randomFloats(Random rng, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (rng.nextFloat() - 0.5f) * 2.0f;
        }
        return out;
    }

    /**
     * BERT-Style LayerNorm: y = (x - μ) / √(σ² + ε) · γ + β über die letzte Dim.
     */
    private static float[] cpuLayerNorm(float[] x, float[] gamma, float[] beta,
                                        int M, int H, float eps) {
        float[] y = new float[M * H];
        for (int m = 0; m < M; m++) {
            int base = m * H;
            double mean = 0.0;
            for (int h = 0; h < H; h++) mean += x[base + h];
            mean /= H;
            double var = 0.0;
            for (int h = 0; h < H; h++) {
                double d = x[base + h] - mean;
                var += d * d;
            }
            var /= H;
            double inv = 1.0 / Math.sqrt(var + eps);
            for (int h = 0; h < H; h++) {
                y[base + h] = (float) ((x[base + h] - mean) * inv * gamma[h] + beta[h]);
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

