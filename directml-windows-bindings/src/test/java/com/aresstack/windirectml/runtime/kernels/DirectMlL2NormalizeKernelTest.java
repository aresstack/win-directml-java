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
 * GPU-Roundtrip-Test für {@link DirectMlL2NormalizeKernel}.
 * <p>
 * Vergleicht den GPU-Pfad {@code y = x / sqrt(Σ x_j² + ε²)} gegen die
 * CPU-Referenzformel {@code y = x / max(sqrt(Σ x_j²), ε)} auf einem
 * deterministischen Float-Input. Für vernünftige Eingangsvektoren (Norm
 * ≫ ε) sind beide Formeln in FP32 ununterscheidbar; die Toleranz ist
 * trotzdem großzügig (1e-5) angesetzt, weil DirectML die Quadratsumme
 * intern in anderer Reihenfolge bildet.
 * <p>
 * Skippt sauber, wenn auf der Maschine kein D3D12/DML-Stack verfügbar ist.
 */
class DirectMlL2NormalizeKernelTest {

    private static final int N = 384;
    private static final float EPSILON = 1e-12f;

    @Test
    void l2NormalizeMatchesCpuReference() throws DirectMlRuntimeException {
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

            Random rng = new Random(0xCAFE);
            float[] xData = randomFloats(rng, N);
            float[] expected = cpuL2Normalize(xData, EPSILON);

            CpuTensor xCpu = CpuTensor.float32(TensorShape.of(N), xData);

            try (GpuBuffer xBuf = ctx.allocateBufferFor(xCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer yBuf = ctx.allocateBuffer((long) N * Float.BYTES,
                         GpuBuffer.BufferUsage.ACTIVATION);
                 DirectMlL2NormalizeKernel kernel = new DirectMlL2NormalizeKernel(ctx, N, EPSILON)) {

                xBuf.upload(xCpu);

                DirectMlTensor x = new DirectMlTensor(TensorShape.of(N),
                        TensorLayout.rowMajor(TensorShape.of(N)),
                        TensorDataType.FLOAT32, xBuf);
                DirectMlTensor y = new DirectMlTensor(TensorShape.of(N),
                        TensorLayout.rowMajor(TensorShape.of(N)),
                        TensorDataType.FLOAT32, yBuf);

                kernel.dispatch(x, y, EPSILON);

                CpuTensor yOut = emptyCpu(N);
                yBuf.download(yOut);
                float[] actual = readFloats(yOut, N);

                assertArrayEquals(expected, actual, 1e-5f,
                        "DirectML L2-normalize must match the CPU reference within FP32 rounding");

                // Sanity: the result must have unit L2 norm.
                double norm = 0.0;
                for (float v : actual) norm += (double) v * v;
                assertTrue(Math.abs(Math.sqrt(norm) - 1.0) < 1e-5,
                        "L2-normalised vector must have unit norm, got " + Math.sqrt(norm));

                // Kernel metadata round-trip.
                assertEquals(N, kernel.elementCount());
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

            float[] xData = new float[N];
            xData[0] = 1.0f;
            CpuTensor xCpu = CpuTensor.float32(TensorShape.of(N), xData);

            try (GpuBuffer xBuf = ctx.allocateBufferFor(xCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer yBuf = ctx.allocateBuffer((long) N * Float.BYTES,
                         GpuBuffer.BufferUsage.ACTIVATION);
                 DirectMlL2NormalizeKernel kernel = new DirectMlL2NormalizeKernel(ctx, N, EPSILON)) {

                xBuf.upload(xCpu);
                DirectMlTensor x = new DirectMlTensor(TensorShape.of(N),
                        TensorLayout.rowMajor(TensorShape.of(N)), TensorDataType.FLOAT32, xBuf);
                DirectMlTensor y = new DirectMlTensor(TensorShape.of(N),
                        TensorLayout.rowMajor(TensorShape.of(N)), TensorDataType.FLOAT32, yBuf);

                // ε is baked at compile time → a different runtime ε must fail.
                assertThrows(DirectMlRuntimeException.class,
                        () -> kernel.dispatch(x, y, 1e-6f));
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

    private static float[] cpuL2Normalize(float[] x, float epsilon) {
        double sum = 0.0;
        for (float v : x) sum += (double) v * v;
        double norm = Math.sqrt(sum);
        double inv = 1.0 / Math.max(norm, epsilon);
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) y[i] = (float) (x[i] * inv);
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

