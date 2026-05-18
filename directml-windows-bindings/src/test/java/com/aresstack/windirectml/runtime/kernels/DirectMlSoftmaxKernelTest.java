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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Verifiziert {@link DirectMlSoftmaxKernel} gegen eine numerisch stabile
 * CPU-Referenz auf einem 2-D-Tensor mit 8 Zeilen × 32 Spalten.
 * <p>
 * {@code DML_OPERATOR_ACTIVATION_SOFTMAX} ist FL 2.0 und steht in jeder
 * heute ausgelieferten {@code DirectML.dll} zur Verfügung – kein
 * Feature-Level-Gating nötig.
 */
class DirectMlSoftmaxKernelTest {

    private static final int ROWS = 8;
    private static final int COLS = 32;

    @Test
    void softmaxMatchesCpuReference() throws DirectMlRuntimeException {
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

            Random rng = new Random(0xDEC0DE);
            float[] xData = randomFloats(rng, ROWS * COLS);
            float[] expected = cpuSoftmax(xData, ROWS, COLS);

            CpuTensor xCpu = CpuTensor.float32(TensorShape.of(ROWS, COLS), xData);
            try (GpuBuffer xBuf = ctx.allocateBufferFor(xCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer yBuf = ctx.allocateBuffer((long) ROWS * COLS * Float.BYTES,
                         GpuBuffer.BufferUsage.ACTIVATION);
                 DirectMlSoftmaxKernel kernel = new DirectMlSoftmaxKernel(ctx, ROWS, COLS)) {

                xBuf.upload(xCpu);

                DirectMlTensor x = tensor2D(xBuf, ROWS, COLS);
                DirectMlTensor y = tensor2D(yBuf, ROWS, COLS);
                kernel.dispatch(x, y);

                CpuTensor yOut = emptyCpu(ROWS * COLS);
                yBuf.download(yOut);
                float[] actual = readFloats(yOut, ROWS * COLS);

                // Toleranz 1e-5: Softmax ist numerisch sehr stabil, und
                // beide Implementierungen subtrahieren das Row-Maximum.
                assertArrayEquals(expected, actual, 1e-5f,
                        "DirectML Softmax must match the CPU reference");

                // Sanity: jede Zeile muss zu 1.0 summieren.
                for (int r = 0; r < ROWS; r++) {
                    float sum = 0;
                    for (int c = 0; c < COLS; c++) sum += actual[r * COLS + c];
                    assertEquals(1.0f, sum, 1e-5f, "row " + r + " must sum to 1");
                }
            }
        } finally {
            ctx.close();
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────

    private static DirectMlTensor tensor2D(GpuBuffer buf, int rows, int cols) {
        TensorShape s = TensorShape.of(rows, cols);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private static float[] randomFloats(Random rng, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            // Range [-5, +5] – außerhalb sättigt softmax und 1e-5 Toleranz
            // wird wegen exp-Vorberechnung schwierig.
            out[i] = (rng.nextFloat() - 0.5f) * 10.0f;
        }
        return out;
    }

    /**
     * Row-weise numerisch stabile Softmax: subtrahiere Zeilenmax vor exp().
     */
    private static float[] cpuSoftmax(float[] x, int rows, int cols) {
        float[] y = new float[x.length];
        for (int r = 0; r < rows; r++) {
            int off = r * cols;
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < cols; c++) max = Math.max(max, x[off + c]);
            double sum = 0;
            for (int c = 0; c < cols; c++) {
                double e = Math.exp(x[off + c] - max);
                y[off + c] = (float) e;
                sum += e;
            }
            float invSum = (float) (1.0 / sum);
            for (int c = 0; c < cols; c++) y[off + c] *= invSum;
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

