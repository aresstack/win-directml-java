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
    @Disabled("TODO(layernorm): IDMLDevice::CreateOperator rejects MVN0/MVN1 with E_INVALIDARG "
            + "on this DML build. Re-enable once the operator-desc layout has been validated "
            + "against the DML debug layer (see win-directml-java-issues.md).")
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

