package com.aresstack.windirectml.encoder.minilm;

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
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Synthetischer Single-Layer-MiniLM-Vergleichstest:
 * {@link DirectMlMiniLmLayerBlock} muss bit-nahe denselben Output liefern
 * wie {@link CpuMiniLmEncoder#forwardSingleLayer} bei identischen
 * Gewichten und identischem Input.
 * <p>
 * Bewusst kleine Form, um Akkumulationsfehler überschaubar zu halten:
 * {@code seq=4, hidden=12, heads=2, headDim=6, intermediate=24}.
 * <p>
 * Toleranz: {@code 2e-3} absolute, {@code 1e-3} relativ.
 * Die Kette hat 6× GEMM + Softmax + 2× LayerNorm + GELU + 3× elementwise
 * Add/Identity – akkumulierte Float-Rundung liegt erfahrungsgemäß bei
 * 5e-4…1e-3. Die Toleranz nutzt den Maximumsabstand.
 * <p>
 * Voraussetzung: {@code DML_FEATURE_LEVEL_5_1} (für die fused GELU).
 * Auf der Windows-11-RTM In-Box DLL (1.8.0, FL 5.0) wird der Test
 * {@code Assumption}-skipped; eine neuere {@code DirectML.dll} kann via
 * {@code -Dwindirectml.directml.dll=...} eingehängt werden.
 */
class DirectMlMiniLmLayerBlockTest {

    private static final int SEQ = 4;
    private static final int HEADS = 2;
    private static final int HEAD_DIM = 6;
    private static final int HIDDEN = HEADS * HEAD_DIM; // 12
    private static final int INTER = 24;
    private static final float EPS = 1e-12f;

    @Test
    void singleLayerMatchesCpuReference() throws DirectMlRuntimeException {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12");

        DirectMlContextImpl ctx = new DirectMlContextImpl("directml");
        try {
            try { ctx.initialize(); }
            catch (DirectMlRuntimeException e) { assumeTrue(false, "no DML: " + e.getMessage()); return; }
            assumeTrue(ctx.isReady() && ctx.bindings().hasDirectMl(),
                    "Skipping: no DirectML device on this adapter");
            int fl = ctx.bindings().getDmlFeatureLevel();
            assumeTrue(DirectMlBindings.supportsFusedGelu(fl),
                    "Skipping: fused GELU requires DML_FEATURE_LEVEL_5_1, "
                            + "DirectML.dll reports " + DirectMlBindings.formatFeatureLevel(fl)
                            + " (set -Dwindirectml.directml.dll to a redistributable to enable)");

            // ── 1. Build synthetic weights ────────────────────────────────
            Random rng = new Random(0xB10C1L);
            float[] qw = rand(rng, HIDDEN * HIDDEN, 0.05f);
            float[] qb = zeros(HIDDEN);
            float[] kw = rand(rng, HIDDEN * HIDDEN, 0.05f);
            float[] kb = zeros(HIDDEN);
            float[] vw = rand(rng, HIDDEN * HIDDEN, 0.05f);
            float[] vb = zeros(HIDDEN);
            float[] ow = rand(rng, HIDDEN * HIDDEN, 0.05f);
            float[] ob = zeros(HIDDEN);
            float[] ag = ones(HIDDEN);
            float[] ab = zeros(HIDDEN);
            float[] iw = rand(rng, INTER * HIDDEN, 0.05f);
            float[] ib = zeros(INTER);
            float[] mw = rand(rng, HIDDEN * INTER, 0.05f);
            float[] mb = zeros(HIDDEN);
            float[] og = ones(HIDDEN);
            float[] ob2 = zeros(HIDDEN);
            CpuMiniLmWeights.LayerWeights cpuLw = CpuMiniLmWeights.layerForTesting(
                    qw, qb, kw, kb, vw, vb, ow, ob, ag, ab, iw, ib, mw, mb, og, ob2);

            // ── 2. CPU reference forward ──────────────────────────────────
            float[] xInit = rand(rng, SEQ * HIDDEN, 0.1f);
            int[] attentionMask = ones1i(SEQ); // no padding
            float[] xCpu = xInit.clone();
            float[] q = new float[SEQ * HIDDEN];
            float[] kk = new float[SEQ * HIDDEN];
            float[] v = new float[SEQ * HIDDEN];
            float[] attn = new float[SEQ * HIDDEN];
            float[] attnOut = new float[SEQ * HIDDEN];
            float[] mlpInter = new float[SEQ * INTER];
            float[] mlpOut = new float[SEQ * HIDDEN];
            float[] scoresBuf = new float[SEQ * SEQ];
            CpuMiniLmEncoder.forwardSingleLayer(xCpu, cpuLw, attentionMask, SEQ,
                    HIDDEN, HEADS, HEAD_DIM, INTER, EPS,
                    q, kk, v, attn, attnOut, mlpInter, mlpOut, scoresBuf);

            // ── 3. DirectML LayerBlock forward ────────────────────────────
            // Mask as float [SEQ]: 0.0 = valid, -1e9 = padded.
            // attentionMask is all ones, so all zeros here.
            float[] maskF = zeros(SEQ);

            CpuTensor xInitCpu = CpuTensor.float32(TensorShape.of(SEQ, HIDDEN), xInit);
            CpuTensor maskCpu  = CpuTensor.float32(TensorShape.of(SEQ), maskF);

            // GPU weight buffers (caller-owned).
            try (GpuBuffer xInBuf  = ctx.allocateBufferFor(xInitCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer mskBuf  = ctx.allocateBufferFor(maskCpu,  GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer xOutBuf = ctx.allocateBuffer((long) SEQ * HIDDEN * Float.BYTES,
                         GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer qwB  = weight(ctx, qw, HIDDEN, HIDDEN);
                 GpuBuffer qbB  = vec(ctx, qb, HIDDEN);
                 GpuBuffer kwB  = weight(ctx, kw, HIDDEN, HIDDEN);
                 GpuBuffer kbB  = vec(ctx, kb, HIDDEN);
                 GpuBuffer vwB  = weight(ctx, vw, HIDDEN, HIDDEN);
                 GpuBuffer vbB  = vec(ctx, vb, HIDDEN);
                 GpuBuffer owB  = weight(ctx, ow, HIDDEN, HIDDEN);
                 GpuBuffer obB  = vec(ctx, ob, HIDDEN);
                 GpuBuffer agB  = vec(ctx, ag, HIDDEN);
                 GpuBuffer abB  = vec(ctx, ab, HIDDEN);
                 GpuBuffer iwB  = weight(ctx, iw, INTER, HIDDEN);
                 GpuBuffer ibB  = vec(ctx, ib, INTER);
                 GpuBuffer mwB  = weight(ctx, mw, HIDDEN, INTER);
                 GpuBuffer mbB  = vec(ctx, mb, HIDDEN);
                 GpuBuffer ogB  = vec(ctx, og, HIDDEN);
                 GpuBuffer ob2B = vec(ctx, ob2, HIDDEN);
                 DirectMlMiniLmLayerBlock block = new DirectMlMiniLmLayerBlock(
                         ctx, SEQ, HIDDEN, HEADS, HEAD_DIM, INTER, EPS, /* hasMask */ true)) {

                xInBuf.upload(xInitCpu);
                mskBuf.upload(maskCpu);

                DirectMlMiniLmLayerBlock.LayerWeights gpuLw = new DirectMlMiniLmLayerBlock.LayerWeights(
                        qwB, qbB, kwB, kbB, vwB, vbB, owB, obB, agB, abB,
                        iwB, ibB, mwB, mbB, ogB, ob2B);

                DirectMlTensor xT    = tensor(xInBuf,  SEQ, HIDDEN);
                DirectMlTensor maskT = tensor(mskBuf, SEQ);
                DirectMlTensor xOutT = tensor(xOutBuf, SEQ, HIDDEN);

                block.dispatch(xT, gpuLw, maskT, xOutT);

                CpuTensor xOutCpu = emptyCpu(SEQ * HIDDEN);
                xOutBuf.download(xOutCpu);
                float[] xGpu = readFloats(xOutCpu, SEQ * HIDDEN);

                // ── 4. Comparison ─────────────────────────────────────
                float maxAbs = 0f, maxRel = 0f;
                for (int i = 0; i < xGpu.length; i++) {
                    float diff = Math.abs(xGpu[i] - xCpu[i]);
                    maxAbs = Math.max(maxAbs, diff);
                    float denom = Math.max(Math.abs(xCpu[i]), 1e-6f);
                    maxRel = Math.max(maxRel, diff / denom);
                }
                System.out.printf("LayerBlock CPU vs DML  maxAbs=%.6f  maxRel=%.6f%n",
                        maxAbs, maxRel);

                assertArrayEquals(xCpu, xGpu, 2e-3f,
                        "DirectML single layer must match CPU reference (abs<=2e-3); "
                                + "maxAbs=" + maxAbs + " maxRel=" + maxRel);
                assertTrue(maxRel < 1e-2f,
                        "max relative error too large: " + maxRel);
            }
        } finally {
            ctx.close();
        }
    }

    // ── small helpers ───────────────────────────────────────────────────

    private static GpuBuffer weight(DirectMlContextImpl ctx, float[] data, int rows, int cols)
            throws DirectMlRuntimeException {
        CpuTensor t = CpuTensor.float32(TensorShape.of(rows, cols), data);
        GpuBuffer b = ctx.allocateBufferFor(t, GpuBuffer.BufferUsage.WEIGHT);
        b.upload(t);
        return b;
    }

    private static GpuBuffer vec(DirectMlContextImpl ctx, float[] data, int n)
            throws DirectMlRuntimeException {
        CpuTensor t = CpuTensor.float32(TensorShape.of(n), data);
        GpuBuffer b = ctx.allocateBufferFor(t, GpuBuffer.BufferUsage.WEIGHT);
        b.upload(t);
        return b;
    }

    private static DirectMlTensor tensor(GpuBuffer buf, int... dims) {
        TensorShape s = TensorShape.of(dims);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private static float[] rand(Random r, int n, float scale) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float) (r.nextGaussian() * scale);
        return a;
    }

    private static float[] zeros(int n) { return new float[n]; }

    private static float[] ones(int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = 1f;
        return a;
    }

    private static int[] ones1i(int n) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) a[i] = 1;
        return a;
    }

    private static CpuTensor emptyCpu(int n) {
        ByteBuffer storage = ByteBuffer.allocateDirect(n * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        TensorShape s = TensorShape.of(n);
        return new CpuTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, storage);
    }

    private static float[] readFloats(CpuTensor t, int n) {
        float[] out = new float[n];
        FloatBuffer fv = t.data().duplicate().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        fv.position(0);
        fv.get(out, 0, n);
        return out;
    }
}

