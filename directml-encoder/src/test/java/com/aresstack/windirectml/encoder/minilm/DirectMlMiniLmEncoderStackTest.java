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
 * Synthetischer Multi-Layer-MiniLM-Vergleichstest:
 * {@link DirectMlMiniLmEncoderStack} muss bit-nahe denselben Output
 * liefern wie {@link CpuMiniLmEncoder#forwardEmbeddingAndLayers} bei
 * identischen Gewichten, identischem Pre-Embedding-Input und identischer
 * Maske.
 * <p>
 * Form bewusst klein gehalten – zwei Layer reichen, um sicherzustellen,
 * dass das Ping-Pong-Buffering zwischen scratchA/scratchB stimmt und der
 * Embedding-LayerNorm-Output korrekt in den ersten Block fließt:
 * {@code seq=4, hidden=12, heads=2, headDim=6, intermediate=24, layers=2}.
 * <p>
 * Toleranz: {@code 3e-3} absolute, {@code 1e-2} relativ.
 * Zwei Layer akkumulieren grob das doppelte Float-Rauschen eines Layers
 * (~5e-4…1e-3), zzgl. eines vorgelagerten LayerNorms. Die Toleranz nutzt
 * den Maximumsabstand.
 */
class DirectMlMiniLmEncoderStackTest {

    private static final int SEQ = 4;
    private static final int HEADS = 2;
    private static final int HEAD_DIM = 6;
    private static final int HIDDEN = HEADS * HEAD_DIM;   // 12
    private static final int INTER = 24;
    private static final int LAYERS = 2;
    private static final float EPS = 1e-12f;

    @Test
    void multiLayerMatchesCpuReference() throws DirectMlRuntimeException {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12");

        DirectMlContextImpl ctx = new DirectMlContextImpl("directml");
        try {
            try { ctx.initialize(); }
            catch (DirectMlRuntimeException e) { assumeTrue(false, "no DML: " + e.getMessage()); return; }
            assumeTrue(ctx.isReady() && ctx.bindings().hasDirectMl(),
                    "Skipping: no DirectML device on this adapter");
            // GELU strategy is auto-selected per feature level (composite fallback on FL<5.1).

            // ── 1. Build synthetic weights ────────────────────────────────
            Random rng = new Random(0xB10C2L);

            float[] embGamma = ones(HIDDEN);
            float[] embBeta  = zeros(HIDDEN);

            List<CpuMiniLmWeights.LayerWeights> cpuLayers = new ArrayList<>(LAYERS);
            // Pro Layer einen eigenen Satz Gewichte, damit Layer 1 nicht zufällig
            // dieselbe Funktion lernt wie Layer 0 (echter Stack-Test).
            float[][] qw = new float[LAYERS][], kw = new float[LAYERS][], vw = new float[LAYERS][];
            float[][] ow = new float[LAYERS][], iw = new float[LAYERS][], mw = new float[LAYERS][];
            float[][] qb = new float[LAYERS][], kb = new float[LAYERS][], vb = new float[LAYERS][];
            float[][] ob = new float[LAYERS][], ib = new float[LAYERS][], mb = new float[LAYERS][];
            float[][] ag = new float[LAYERS][], ab = new float[LAYERS][];
            float[][] og = new float[LAYERS][], ob2 = new float[LAYERS][];
            for (int l = 0; l < LAYERS; l++) {
                qw[l] = rand(rng, HIDDEN * HIDDEN, 0.05f); qb[l] = zeros(HIDDEN);
                kw[l] = rand(rng, HIDDEN * HIDDEN, 0.05f); kb[l] = zeros(HIDDEN);
                vw[l] = rand(rng, HIDDEN * HIDDEN, 0.05f); vb[l] = zeros(HIDDEN);
                ow[l] = rand(rng, HIDDEN * HIDDEN, 0.05f); ob[l] = zeros(HIDDEN);
                ag[l] = ones(HIDDEN);                      ab[l] = zeros(HIDDEN);
                iw[l] = rand(rng, INTER * HIDDEN, 0.05f);  ib[l] = zeros(INTER);
                mw[l] = rand(rng, HIDDEN * INTER, 0.05f);  mb[l] = zeros(HIDDEN);
                og[l] = ones(HIDDEN);                      ob2[l] = zeros(HIDDEN);
                cpuLayers.add(CpuMiniLmWeights.layerForTesting(
                        qw[l], qb[l], kw[l], kb[l], vw[l], vb[l],
                        ow[l], ob[l], ag[l], ab[l],
                        iw[l], ib[l], mw[l], mb[l], og[l], ob2[l]));
            }

            // ── 2. CPU reference: Embedding-LN + N Layer ─────────────────
            float[] xInit = rand(rng, SEQ * HIDDEN, 0.1f); // schon summierte Embeddings
            int[] attentionMask = ones1i(SEQ);
            float[] xCpu = xInit.clone();
            float[] q = new float[SEQ * HIDDEN];
            float[] kk = new float[SEQ * HIDDEN];
            float[] v = new float[SEQ * HIDDEN];
            float[] attn = new float[SEQ * HIDDEN];
            float[] attnOut = new float[SEQ * HIDDEN];
            float[] mlpInter = new float[SEQ * INTER];
            float[] mlpOut = new float[SEQ * HIDDEN];
            float[] scoresBuf = new float[SEQ * SEQ];
            CpuMiniLmEncoder.forwardEmbeddingAndLayers(xCpu, embGamma, embBeta, cpuLayers,
                    attentionMask, SEQ, HIDDEN, HEADS, HEAD_DIM, INTER, EPS,
                    q, kk, v, attn, attnOut, mlpInter, mlpOut, scoresBuf);

            // ── 3. DirectML stack forward ────────────────────────────────
            float[] maskF = zeros(SEQ);  // all valid → 0.0

            CpuTensor xInitCpu = CpuTensor.float32(TensorShape.of(SEQ, HIDDEN), xInit);
            CpuTensor maskCpu  = CpuTensor.float32(TensorShape.of(SEQ), maskF);

            // Weight uploads – outer try-with to manage lifetimes uniformly.
            List<GpuBuffer> ownedWeights = new ArrayList<>();
            try (GpuBuffer xInBuf  = ctx.allocateBufferFor(xInitCpu, GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer mskBuf  = ctx.allocateBufferFor(maskCpu,  GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer xOutBuf = ctx.allocateBuffer((long) SEQ * HIDDEN * Float.BYTES,
                         GpuBuffer.BufferUsage.ACTIVATION);
                 GpuBuffer embGammaB = vec(ctx, embGamma, HIDDEN);
                 GpuBuffer embBetaB  = vec(ctx, embBeta, HIDDEN);
                 DirectMlMiniLmEncoderStack stack = new DirectMlMiniLmEncoderStack(
                         ctx, SEQ, HIDDEN, HEADS, HEAD_DIM, INTER, LAYERS, EPS, /* hasMask */ true)) {

                ownedWeights.add(embGammaB); ownedWeights.add(embBetaB);

                xInBuf.upload(xInitCpu);
                mskBuf.upload(maskCpu);

                List<DirectMlMiniLmLayerBlock.LayerWeights> gpuLayers = new ArrayList<>(LAYERS);
                for (int l = 0; l < LAYERS; l++) {
                    GpuBuffer qwB  = weight(ctx, qw[l], HIDDEN, HIDDEN);
                    GpuBuffer qbB  = vec(ctx, qb[l], HIDDEN);
                    GpuBuffer kwB  = weight(ctx, kw[l], HIDDEN, HIDDEN);
                    GpuBuffer kbB  = vec(ctx, kb[l], HIDDEN);
                    GpuBuffer vwB  = weight(ctx, vw[l], HIDDEN, HIDDEN);
                    GpuBuffer vbB  = vec(ctx, vb[l], HIDDEN);
                    GpuBuffer owB  = weight(ctx, ow[l], HIDDEN, HIDDEN);
                    GpuBuffer obB  = vec(ctx, ob[l], HIDDEN);
                    GpuBuffer agB  = vec(ctx, ag[l], HIDDEN);
                    GpuBuffer abB  = vec(ctx, ab[l], HIDDEN);
                    GpuBuffer iwB  = weight(ctx, iw[l], INTER, HIDDEN);
                    GpuBuffer ibB  = vec(ctx, ib[l], INTER);
                    GpuBuffer mwB  = weight(ctx, mw[l], HIDDEN, INTER);
                    GpuBuffer mbB  = vec(ctx, mb[l], HIDDEN);
                    GpuBuffer ogB  = vec(ctx, og[l], HIDDEN);
                    GpuBuffer ob2B = vec(ctx, ob2[l], HIDDEN);
                    ownedWeights.add(qwB); ownedWeights.add(qbB);
                    ownedWeights.add(kwB); ownedWeights.add(kbB);
                    ownedWeights.add(vwB); ownedWeights.add(vbB);
                    ownedWeights.add(owB); ownedWeights.add(obB);
                    ownedWeights.add(agB); ownedWeights.add(abB);
                    ownedWeights.add(iwB); ownedWeights.add(ibB);
                    ownedWeights.add(mwB); ownedWeights.add(mbB);
                    ownedWeights.add(ogB); ownedWeights.add(ob2B);
                    gpuLayers.add(new DirectMlMiniLmLayerBlock.LayerWeights(
                            qwB, qbB, kwB, kbB, vwB, vbB, owB, obB, agB, abB,
                            iwB, ibB, mwB, mbB, ogB, ob2B));
                }

                DirectMlTensor xT       = tensor(xInBuf,    SEQ, HIDDEN);
                DirectMlTensor embGT    = tensor(embGammaB, HIDDEN);
                DirectMlTensor embBT    = tensor(embBetaB,  HIDDEN);
                DirectMlTensor maskT    = tensor(mskBuf,    SEQ);
                DirectMlTensor xOutT    = tensor(xOutBuf,   SEQ, HIDDEN);

                stack.dispatch(xT, embGT, embBT, gpuLayers, maskT, xOutT);

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
                System.out.printf("EncoderStack(L=%d) CPU vs DML  maxAbs=%.6f  maxRel=%.6f%n",
                        LAYERS, maxAbs, maxRel);

                assertArrayEquals(xCpu, xGpu, 3e-3f,
                        "DirectML " + LAYERS + "-layer stack must match CPU reference (abs<=3e-3); "
                                + "maxAbs=" + maxAbs + " maxRel=" + maxRel);
                assertTrue(maxRel < 1e-2f,
                        "max relative error too large: " + maxRel);
            } finally {
                for (int i = ownedWeights.size() - 1; i >= 0; i--) {
                    try { ownedWeights.get(i).close(); } catch (Exception ignored) {}
                }
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

