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
 * Verifiziert {@link DirectMlCompositeGeluKernel} ({@code ERF + IDENTITY
 * + MULTIPLY}, FL 2.0-Fallback) gegen dieselbe exakte CPU-Referenz
 * {@code 0.5·x·(1 + erf(x/√2))}, die auch {@code DirectMlGeluKernel}
 * matcht.
 * <p>
 * Wichtig: dieser Test ist <em>unabhängig</em> vom Feature-Level – er
 * instanziiert die Composite-Variante <em>immer</em> direkt. Damit ist
 * der Fallback-Pfad auch auf modernen DLLs (FL 6.4) als Regressionstest
 * gesichert, nicht nur auf alten In-Box-DLLs.
 */
class DirectMlCompositeGeluKernelTest {

    private static final int N = 256;

    @Test
    void compositeGeluMatchesCpuReference() throws DirectMlRuntimeException {
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
            float[] xData = randomFloats(rng, N);
            float[] expected = cpuGelu(xData);

            CpuTensor xCpu = CpuTensor.float32(TensorShape.of(N), xData);
            try (GpuBuffer xBuf = ctx.allocateBufferFor(xCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer yBuf = ctx.allocateBuffer((long) N * Float.BYTES,
                         GpuBuffer.BufferUsage.ACTIVATION);
                 DirectMlCompositeGeluKernel kernel = new DirectMlCompositeGeluKernel(ctx, N)) {

                xBuf.upload(xCpu);

                DirectMlTensor x = tensor1D(xBuf, N);
                DirectMlTensor y = tensor1D(yBuf, N);
                kernel.dispatch(x, y);

                CpuTensor yOut = emptyCpu(N);
                yBuf.download(yOut);
                float[] actual = readFloats(yOut, N);

                // Composite path uses DirectML's native ERF approximation
                // (different from A&S 7.1.26) so the gap to CPU is slightly
                // larger than the fused path. 5e-4 absolute is comfortable
                // for the [-4, 4] input range without false positives.
                assertArrayEquals(expected, actual, 5e-4f,
                        "Composite GELU (ERF+IDENTITY+MUL) must match the CPU reference");
            }
        } finally {
            ctx.close();
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────

    private static DirectMlTensor tensor1D(GpuBuffer buf, int n) {
        TensorShape s = TensorShape.of(n);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private static float[] randomFloats(Random rng, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (rng.nextFloat() - 0.5f) * 8.0f;
        }
        return out;
    }

    private static float[] cpuGelu(float[] x) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            float xi = x[i];
            y[i] = (float) (0.5 * xi * (1.0 + erf(xi / Math.sqrt(2.0))));
        }
        return y;
    }

    private static double erf(double xRaw) {
        double sign = xRaw < 0 ? -1.0 : 1.0;
        double x = Math.abs(xRaw);
        double t = 1.0 / (1.0 + 0.3275911 * x);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
        return sign * y;
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

