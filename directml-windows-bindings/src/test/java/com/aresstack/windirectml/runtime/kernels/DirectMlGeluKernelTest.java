package com.aresstack.windirectml.runtime.kernels;

import com.aresstack.windirectml.runtime.CpuTensor;
import com.aresstack.windirectml.runtime.DirectMlContextImpl;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;
import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.GpuBuffer;
import com.aresstack.windirectml.runtime.TensorDataType;
import com.aresstack.windirectml.runtime.TensorLayout;
import com.aresstack.windirectml.runtime.TensorShape;
import com.aresstack.windirectml.windows.DirectMlBindings;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Verifiziert {@link DirectMlGeluKernel} ({@code DML_OPERATOR_ACTIVATION_GELU},
 * Enum-ID 157) gegen die exakte CPU-Referenz
 * {@code 0.5·x·(1 + erf(x/√2))} – dieselbe Formel, die
 * {@code CpuMiniLmEncoder} im FFN-Block verwendet.
 */
class DirectMlGeluKernelTest {

    private static final int N = 256;

    @Test
    void geluMatchesCpuReference() throws DirectMlRuntimeException {
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

            // DML_OPERATOR_ACTIVATION_GELU (op 157) requires DML_FEATURE_LEVEL_5_1.
            // The in-box DirectML.dll shipped with Windows 11 21H2/22H2 RTM is
            // v1.8.0 (FL 5.0). Bundle a Microsoft.AI.DirectML redistributable
            // (>= 1.10) or point -Dwindirectml.directml.dll at one to enable
            // the fused kernel. A composite (ERF + IDENTITY + MUL) fallback
            // is tracked as a follow-up sprint.
            int fl = ctx.bindings().getDmlFeatureLevel();
            assumeTrue(DirectMlBindings.supportsFusedGelu(fl),
                    "Skipping: fused GELU requires DML_FEATURE_LEVEL_5_1, but " +
                            "DirectML.dll reports " + DirectMlBindings.formatFeatureLevel(fl) +
                            " (set -Dwindirectml.directml.dll to a bundled redist)");

            Random rng = new Random(0xC0FFEE);
            float[] xData = randomFloats(rng, N);
            float[] expected = cpuGelu(xData);

            CpuTensor xCpu = CpuTensor.float32(TensorShape.of(N), xData);
            try (GpuBuffer xBuf = ctx.allocateBufferFor(xCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer yBuf = ctx.allocateBuffer((long) N * Float.BYTES,
                         GpuBuffer.BufferUsage.ACTIVATION);
                 DirectMlGeluKernel kernel = new DirectMlGeluKernel(ctx, N)) {

                xBuf.upload(xCpu);

                DirectMlTensor x = tensor1D(xBuf, N);
                DirectMlTensor y = tensor1D(yBuf, N);
                kernel.dispatch(x, y);

                CpuTensor yOut = emptyCpu(N);
                yBuf.download(yOut);
                float[] actual = readFloats(yOut, N);

                // GELU ist analytisch glatt; 1e-4 ist ausreichend für die
                // erf-Approximationen, die DirectML und die CPU-Referenz
                // unabhängig voneinander wählen.
                assertArrayEquals(expected, actual, 1e-4f,
                        "DirectML GELU must match the CPU reference");
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
            // GELU ist über [-4, +4] besonders interessant (Aktivierung
            // erreicht dort ihre nichtlineare Region). Außerhalb sättigt
            // die Funktion gegen 0 bzw. die Identität.
            out[i] = (rng.nextFloat() - 0.5f) * 8.0f;
        }
        return out;
    }

    /**
     * Exakte GELU (BERT-Standard): {@code 0.5 · x · (1 + erf(x / √2))}.
     * Identisch zur Implementierung in {@code CpuMiniLmEncoder#geluInPlace}.
     */
    private static float[] cpuGelu(float[] x) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            float xi = x[i];
            y[i] = (float) (0.5 * xi * (1.0 + erf(xi / Math.sqrt(2.0))));
        }
        return y;
    }

    /**
     * Abramowitz &amp; Stegun 7.1.26 – relativer Fehler &lt; 1.5e-7.
     */
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

