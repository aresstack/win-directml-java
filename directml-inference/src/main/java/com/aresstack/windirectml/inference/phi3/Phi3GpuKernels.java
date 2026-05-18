package com.aresstack.windirectml.inference.phi3;

import com.aresstack.windirectml.inference.phi3.Phi3Weights.LayerWeights;
import com.aresstack.windirectml.inference.phi3.Phi3Weights.QuantizedWeight;
import com.aresstack.windirectml.windows.MatMulNBitsKernel;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * GPU kernel pool for the Phi-3 decoder.
 * <p>
 * <b>V1.2 QKV fusion</b>: Q, K, V projections are fused into a single
 * {@code [3×hidden, hidden]} GEMM per layer, reducing GPU submissions
 * from 6 to 4 per layer (saves 64 fence waits per token for Phi-3-mini).
 * <p>
 * Creates one fused QKV kernel + O/gate_up/down kernels per layer,
 * plus an optional lm_head kernel. Each kernel dequantizes INT4→FP32
 * at creation time and keeps the weight matrix resident on GPU.
 * <p>
 * <b>GPU memory budget (Phi-3-mini, FP32 dequantized):</b>
 * <pre>
 *   qkv fused:  9216 × 3072 × 4 bytes     ≈ 113 MB  (was 3 × 37.7 MB)
 *   o:          3072² × 4                  ≈  37.7 MB
 *   gate_up:    16384 × 3072 × 4           ≈ 201 MB
 *   down:       3072 × 8192 × 4            ≈ 101 MB
 *   ─────────────────────────────────────────────
 *   Per layer:                              ≈ 453 MB  (unchanged)
 *   32 layers:                              ≈ 14.5 GB
 *   lm_head:    32064 × 3072 × 4           ≈ 394 MB
 * </pre>
 */
public final class Phi3GpuKernels implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(Phi3GpuKernels.class);

    // Projection indices within layerKernels[l]
    static final int QKV_FUSED   = 0;
    static final int O_PROJ      = 1;
    static final int GATE_UP_PROJ = 2;
    static final int DOWN_PROJ   = 3;
    static final int PROJECTIONS_PER_LAYER = 4;

    private final int gpuLayers;
    private final MatMulNBitsKernel[][] layerKernels;   // [layer][projection]
    private final MatMulNBitsKernel lmHeadKernel;       // nullable
    private boolean closed = false;

    private Phi3GpuKernels(int gpuLayers,
                           MatMulNBitsKernel[][] layerKernels,
                           MatMulNBitsKernel lmHeadKernel) {
        this.gpuLayers = gpuLayers;
        this.layerKernels = layerKernels;
        this.lmHeadKernel = lmHeadKernel;
    }

    // ── Factory ──────────────────────────────────────────────────────────

    /**
     * Create GPU kernels for the first {@code gpuLayers} decoder layers
     * and optionally the lm_head projection.
     */
    public static Phi3GpuKernels create(WindowsBindings wb, Phi3Weights weights,
                                         Phi3Config config, int gpuLayers,
                                         boolean gpuLmHead) {
        int layers = Math.min(Math.max(gpuLayers, 0), config.numHiddenLayers());
        log.info("Creating GPU kernels: {} / {} layers (QKV fused), lmHead={}",
                layers, config.numHiddenLayers(), gpuLmHead);
        long t0 = System.currentTimeMillis();

        int hidden = config.hiddenSize();
        MatMulNBitsKernel[][] kernels = new MatMulNBitsKernel[layers][];

        for (int l = 0; l < layers; l++) {
            LayerWeights lw = weights.layers[l];
            MatMulNBitsKernel[] k = new MatMulNBitsKernel[PROJECTIONS_PER_LAYER];

            // ── Fused QKV: dequantize Q/K/V separately, concatenate, upload once ──
            k[QKV_FUSED] = createFusedQKV(wb, lw, hidden, l);

            // ── Remaining projections ──
            k[O_PROJ]       = createKernel(wb, lw.oProj(),      "layer." + l + ".o_proj");
            k[GATE_UP_PROJ] = createKernel(wb, lw.gateUpProj(), "layer." + l + ".gate_up");
            k[DOWN_PROJ]    = createKernel(wb, lw.downProj(),   "layer." + l + ".down");
            kernels[l] = k;

            if ((l + 1) % 4 == 0 || l == layers - 1) {
                log.info("GPU kernels: {}/{} layers created ({} ms)",
                        l + 1, layers, System.currentTimeMillis() - t0);
            }
        }

        MatMulNBitsKernel lmHead = null;
        if (gpuLmHead) {
            lmHead = createKernel(wb, weights.lmHead, "lm_head");
            log.info("GPU lm_head kernel created [{}, {}]",
                    weights.lmHead.N(), weights.lmHead.K());
        }

        long elapsed = System.currentTimeMillis() - t0;
        int totalKernels = layers * PROJECTIONS_PER_LAYER + (lmHead != null ? 1 : 0);
        log.info("Phi3GpuKernels ready: {} kernels in {} ms ({} ms/kernel, QKV fused)",
                totalKernels, elapsed, totalKernels > 0 ? elapsed / totalKernels : 0);

        return new Phi3GpuKernels(layers, kernels, lmHead);
    }

    /**
     * Create a fused Q+K+V kernel: dequantize each separately, concatenate
     * into a single [3*hidden, hidden] weight matrix, upload once.
     * <p>
     * Reduces 3 GPU submissions → 1 per layer (saves 2 fence waits).
     */
    private static MatMulNBitsKernel createFusedQKV(WindowsBindings wb,
                                                     LayerWeights lw,
                                                     int hidden, int layerIdx) {
        QuantizedWeight qw = lw.qProj();
        QuantizedWeight kw = lw.kProj();
        QuantizedWeight vw = lw.vProj();

        float[] qDeq = MatMulNBitsKernel.dequantizeInt4(
                qw.qWeight(), qw.scales(), qw.zeroPoints(), qw.N(), qw.K(), qw.blockSize());
        float[] kDeq = MatMulNBitsKernel.dequantizeInt4(
                kw.qWeight(), kw.scales(), kw.zeroPoints(), kw.N(), kw.K(), kw.blockSize());
        float[] vDeq = MatMulNBitsKernel.dequantizeInt4(
                vw.qWeight(), vw.scales(), vw.zeroPoints(), vw.N(), vw.K(), vw.blockSize());

        // Concatenate: [Q; K; V] → [3*hidden, hidden] row-major
        int fusedN = 3 * hidden;
        float[] fused = new float[fusedN * hidden];
        System.arraycopy(qDeq, 0, fused, 0,              hidden * hidden);
        System.arraycopy(kDeq, 0, fused, hidden * hidden,     hidden * hidden);
        System.arraycopy(vDeq, 0, fused, 2 * hidden * hidden, hidden * hidden);

        log.debug("Creating fused QKV kernel layer.{}: [{}, {}]", layerIdx, fusedN, hidden);
        return MatMulNBitsKernel.fromDequantizedWeights(wb, fusedN, hidden, fused);
    }

    private static MatMulNBitsKernel createKernel(WindowsBindings wb,
                                                   QuantizedWeight qw,
                                                   String name) {
        log.debug("Creating GPU kernel: {} [{}, {}]", name, qw.N(), qw.K());
        return new MatMulNBitsKernel(wb, qw.N(), qw.K(),
                qw.qWeight(), qw.scales(), qw.zeroPoints(), qw.blockSize());
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /** Whether layer {@code layerIdx} has GPU kernels. */
    public boolean hasLayer(int layerIdx) {
        return layerIdx >= 0 && layerIdx < gpuLayers;
    }

    /** Whether lm_head has a GPU kernel. */
    public boolean hasLmHead() { return lmHeadKernel != null; }

    /** Number of decoder layers on GPU. */
    public int getGpuLayers() { return gpuLayers; }

    /** Fused Q+K+V kernel: output is [3×hidden], split into Q[0..h), K[h..2h), V[2h..3h). */
    public MatMulNBitsKernel qkvFused(int layer) { return layerKernels[layer][QKV_FUSED]; }
    public MatMulNBitsKernel oProj(int layer)      { return layerKernels[layer][O_PROJ]; }
    public MatMulNBitsKernel gateUpProj(int layer) { return layerKernels[layer][GATE_UP_PROJ]; }
    public MatMulNBitsKernel downProj(int layer)   { return layerKernels[layer][DOWN_PROJ]; }
    public MatMulNBitsKernel lmHead()              { return lmHeadKernel; }

    // ── AutoCloseable ────────────────────────────────────────────────────

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        int count = 0;
        for (MatMulNBitsKernel[] layer : layerKernels) {
            if (layer == null) continue;
            for (MatMulNBitsKernel k : layer) {
                if (k != null) { k.close(); count++; }
            }
        }
        if (lmHeadKernel != null) { lmHeadKernel.close(); count++; }

        log.info("Phi3GpuKernels closed ({} kernels, {} layers)", count, gpuLayers);
    }
}

