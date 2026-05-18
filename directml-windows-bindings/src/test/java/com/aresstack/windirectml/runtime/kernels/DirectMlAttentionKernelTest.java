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
 * Verifiziert {@link DirectMlAttentionKernel} (Composite-SDPA) gegen eine
 * numerisch stabile CPU-Referenz auf einer kleinen festen Geometrie:
 * {@code B=1, H=2, S=4, D=8}.
 * <p>
 * Zwei Szenarien:
 * <ul>
 *   <li>{@code attention_no_mask} – reine SDPA</li>
 *   <li>{@code attention_with_mask} – additive Mask {@code [B, S]}
 *       (jede Batch-Spalte hat genau eine Padding-Position mit
 *       {@code -1e9f})</li>
 * </ul>
 * Toleranz {@code 5e-4} – die Composite-Kette akkumuliert Rundungsfehler
 * aus zwei batched-GEMMs und einer Softmax. Alle Sub-Operatoren sind
 * FL-2.0 und müssen auf jeder ausgelieferten {@code DirectML.dll}
 * grün laufen.
 */
class DirectMlAttentionKernelTest {

    private static final int B = 1;
    private static final int H = 2;
    private static final int S = 4;
    private static final int D = 8;
    private static final float SCALE = (float) (1.0 / Math.sqrt(D));
    private static final float TOLERANCE = 5e-4f;

    @Test
    void attentionMatchesCpuReferenceWithoutMask() throws DirectMlRuntimeException {
        runAttentionCase(false);
    }

    @Test
    void attentionMatchesCpuReferenceWithMask() throws DirectMlRuntimeException {
        runAttentionCase(true);
    }

    private void runAttentionCase(boolean withMask) throws DirectMlRuntimeException {
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

            Random rng = new Random(withMask ? 0xA77E47L : 0xDEC0DEL);
            float[] q = randomFloats(rng, B * H * S * D);
            float[] k = randomFloats(rng, B * H * S * D);
            float[] v = randomFloats(rng, B * H * S * D);
            float[] mask = withMask ? buildMask(B, S, rng) : null;
            float[] expected = cpuAttention(q, k, v, mask, SCALE);

            CpuTensor qCpu = CpuTensor.float32(TensorShape.of(B, H, S, D), q);
            CpuTensor kCpu = CpuTensor.float32(TensorShape.of(B, H, S, D), k);
            CpuTensor vCpu = CpuTensor.float32(TensorShape.of(B, H, S, D), v);
            CpuTensor maskCpu = withMask
                    ? CpuTensor.float32(TensorShape.of(B, S), mask) : null;

            try (GpuBuffer qBuf = ctx.allocateBufferFor(qCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer kBuf = ctx.allocateBufferFor(kCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer vBuf = ctx.allocateBufferFor(vCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer mBuf = (maskCpu != null)
                         ? ctx.allocateBufferFor(maskCpu, GpuBuffer.BufferUsage.ACTIVATION)
                         : null;
                 GpuBuffer yBuf = ctx.allocateBuffer((long) B * H * S * D * Float.BYTES,
                         GpuBuffer.BufferUsage.ACTIVATION);
                 DirectMlAttentionKernel kernel =
                         new DirectMlAttentionKernel(ctx, B, H, S, D, SCALE, withMask)) {

                qBuf.upload(qCpu);
                kBuf.upload(kCpu);
                vBuf.upload(vCpu);
                if (mBuf != null) mBuf.upload(maskCpu);

                DirectMlTensor qT = tensor(qBuf, B, H, S, D);
                DirectMlTensor kT = tensor(kBuf, B, H, S, D);
                DirectMlTensor vT = tensor(vBuf, B, H, S, D);
                DirectMlTensor mT = (mBuf != null) ? tensor(mBuf, B, S) : null;
                DirectMlTensor yT = tensor(yBuf, B, H, S, D);

                kernel.dispatch(qT, kT, vT, mT, yT, SCALE);

                CpuTensor yOut = emptyCpu(B * H * S * D);
                yBuf.download(yOut);
                float[] actual = readFloats(yOut, B * H * S * D);

                assertArrayEquals(expected, actual, TOLERANCE,
                        "DirectML attention must match CPU reference"
                                + (withMask ? " (with mask)" : " (no mask)"));
            }
        } finally {
            ctx.close();
        }
    }

    // ── CPU reference ───────────────────────────────────────────────────

    /**
     * Plain scaled-dot-product attention over {@code [B,H,S,D]}.
     * {@code mask} (optional) is {@code [B,S]} additive – broadcast over
     * H and over the query axis.
     */
    private static float[] cpuAttention(float[] q, float[] k, float[] v,
                                        float[] mask, float scale) {
        float[] y = new float[B * H * S * D];
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {
                // 1. scores[i,j] = scale · Σ_d q[i,d] · k[j,d]
                float[][] scores = new float[S][S];
                for (int i = 0; i < S; i++) {
                    for (int j = 0; j < S; j++) {
                        double s = 0.0;
                        for (int d = 0; d < D; d++) {
                            s += q[idx(b, h, i, d)] * k[idx(b, h, j, d)];
                        }
                        scores[i][j] = (float) (s * scale);
                        if (mask != null) {
                            scores[i][j] += mask[b * S + j];
                        }
                    }
                }
                // 2. softmax row-wise (numerisch stabil)
                float[][] probs = new float[S][S];
                for (int i = 0; i < S; i++) {
                    float max = Float.NEGATIVE_INFINITY;
                    for (int j = 0; j < S; j++) max = Math.max(max, scores[i][j]);
                    double sum = 0.0;
                    for (int j = 0; j < S; j++) {
                        double e = Math.exp(scores[i][j] - max);
                        probs[i][j] = (float) e;
                        sum += e;
                    }
                    float inv = (float) (1.0 / sum);
                    for (int j = 0; j < S; j++) probs[i][j] *= inv;
                }
                // 3. y[i,d] = Σ_j probs[i,j] · v[j,d]
                for (int i = 0; i < S; i++) {
                    for (int d = 0; d < D; d++) {
                        double s = 0.0;
                        for (int j = 0; j < S; j++) {
                            s += probs[i][j] * v[idx(b, h, j, d)];
                        }
                        y[idx(b, h, i, d)] = (float) s;
                    }
                }
            }
        }
        return y;
    }

    private static int idx(int b, int h, int s, int d) {
        return ((b * H + h) * S + s) * D + d;
    }

    // ── helpers ─────────────────────────────────────────────────────────

    private static DirectMlTensor tensor(GpuBuffer buf, int... dims) {
        TensorShape s = TensorShape.of(dims);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private static float[] randomFloats(Random rng, int n) {
        float[] out = new float[n];
        // Skala so wählen, dass scaled scores im moderaten Bereich liegen
        // (sonst sättigt softmax und die Toleranz wird unrealistisch).
        for (int i = 0; i < n; i++) out[i] = (rng.nextFloat() - 0.5f) * 2.0f;
        return out;
    }

    /**
     * Ein einfacher Padding-Mask-Vektor pro Batch: 0.0f für valide
     * Positionen, -1e9f für genau eine geblockte Position pro Batch.
     */
    private static float[] buildMask(int batch, int seq, Random rng) {
        float[] m = new float[batch * seq];
        for (int b = 0; b < batch; b++) {
            int blocked = rng.nextInt(seq);
            for (int s = 0; s < seq; s++) {
                m[b * seq + s] = (s == blocked) ? -1e9f : 0.0f;
            }
        }
        return m;
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

