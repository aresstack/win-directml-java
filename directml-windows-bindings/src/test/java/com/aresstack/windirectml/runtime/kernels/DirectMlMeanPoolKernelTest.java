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
 * GPU-Roundtrip-Test für {@link DirectMlMeanPoolKernel}.
 * <p>
 * Vergleicht {@code y[h] = Σ_t w[t] · x[t,h]} (GEMM-Pfad) gegen eine naive
 * CPU-Referenz auf demselben deterministischen Float-Input. Die Toleranz
 * ist großzügig (1e-4), weil DirectML intern in anderer Reihenfolge
 * summiert.
 * <p>
 * Skippt sauber, wenn auf der Maschine kein D3D12/DML-Stack verfügbar ist.
 */
class DirectMlMeanPoolKernelTest {

    private static final int S = 17;   // bewusst nicht 2-Potenz, um Pad-Logik zu prüfen
    private static final int H = 32;

    @Test
    void meanPoolMatchesCpuReference() throws DirectMlRuntimeException {
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

            Random rng = new Random(0xBEEF);
            float[] xData = randomFloats(rng, S * H);

            // Realistische Mask: 12 echte Tokens, 5 gepadded. Normalisiert.
            int validCount = 12;
            float[] wData = new float[S];
            float inv = 1.0f / validCount;
            for (int t = 0; t < validCount; t++) wData[t] = inv;

            float[] expected = cpuMeanPool(xData, wData, S, H);

            CpuTensor xCpu = CpuTensor.float32(TensorShape.of(S, H), xData);
            CpuTensor wCpu = CpuTensor.float32(TensorShape.of(S), wData);

            try (GpuBuffer xBuf = ctx.allocateBufferFor(xCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer wBuf = ctx.allocateBufferFor(wCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer yBuf = ctx.allocateBuffer((long) H * Float.BYTES,
                         GpuBuffer.BufferUsage.ACTIVATION);
                 DirectMlMeanPoolKernel kernel = new DirectMlMeanPoolKernel(ctx, S, H)) {

                xBuf.upload(xCpu);
                wBuf.upload(wCpu);

                DirectMlTensor x = new DirectMlTensor(TensorShape.of(S, H),
                        TensorLayout.rowMajor(TensorShape.of(S, H)),
                        TensorDataType.FLOAT32, xBuf);
                DirectMlTensor w = new DirectMlTensor(TensorShape.of(S),
                        TensorLayout.rowMajor(TensorShape.of(S)),
                        TensorDataType.FLOAT32, wBuf);
                DirectMlTensor y = new DirectMlTensor(TensorShape.of(H),
                        TensorLayout.rowMajor(TensorShape.of(H)),
                        TensorDataType.FLOAT32, yBuf);

                kernel.dispatch(x, w, y);

                CpuTensor yOut = emptyCpu(H);
                yBuf.download(yOut);
                float[] actual = readFloats(yOut, H);

                assertArrayEquals(expected, actual, 1e-4f,
                        "DirectML mean-pool must match the CPU reference for y[h] = Σ_t w[t]·x[t,h]");
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

    private static float[] cpuMeanPool(float[] x, float[] w, int S, int H) {
        float[] y = new float[H];
        for (int t = 0; t < S; t++) {
            float wt = w[t];
            if (wt == 0f) continue;
            int base = t * H;
            for (int h = 0; h < H; h++) y[h] += wt * x[base + h];
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

