package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.windows.MatMulNBitsKernel;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * GPU kernel pool for the Qwen2 decoder.
 *
 * <p>Creates one fused QKV kernel, o_proj kernel, fused gate+up kernel,
 * and down_proj kernel per decoder layer, plus an optional lm_head kernel.
 * Each kernel dequantizes INT4→FP32 at build time (for quantized models) or
 * copies FP32 directly (for dense FP16/FP32 models) and keeps the weight
 * matrix resident on GPU via DirectML / D3D12.
 *
 * <h2>Fusions</h2>
 * <ul>
 *   <li><b>QKV fused</b>: Q, K, V projections are concatenated into a single
 *       {@code [qSize+2*kvSize, hidden]} weight matrix.  For GQA models
 *       (numAttentionHeads ≠ numKeyValueHeads), Q rows outnumber K/V rows.
 *       Reduces 3 GPU dispatches → 1 per layer.</li>
 *   <li><b>gate+up fused</b>: gate_proj and up_proj are concatenated into a
 *       single {@code [2*intermediateSize, hidden]} weight matrix, matching
 *       the Phi-3 fused gate_up_proj convention. Reduces 2 dispatches → 1.</li>
 * </ul>
 *
 * <h2>GPU memory budget (Qwen2.5-Coder-0.5B, FP32 dequantized)</h2>
 * <pre>
 *   qkv fused:  1152 × 896 × 4          ≈   4.1 MB  per layer
 *   o_proj:      896 × 896 × 4          ≈   3.2 MB  per layer
 *   gate+up:    9728 × 896 × 4          ≈  34.9 MB  per layer
 *   down_proj:   896 × 4864 × 4         ≈  17.5 MB  per layer
 *   ─────────────────────────────────────────────────────────
 *   Per layer:                          ≈  59.7 MB
 *   24 layers:                          ≈  1.4 GB
 *   lm_head: 151936 × 896 × 4          ≈ 544 MB  (optional)
 * </pre>
 */
public final class QwenGpuKernels implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(QwenGpuKernels.class);

    // Kernel indices within layerKernels[l]
    static final int QKV_FUSED = 0;
    static final int O_PROJ = 1;
    static final int GATE_UP_FUSED = 2;
    static final int DOWN_PROJ = 3;
    static final int KERNELS_PER_LAYER = 4;

    private final int gpuLayers;
    private final MatMulNBitsKernel[][] layerKernels;  // [layer][kernel]
    private final MatMulNBitsKernel lmHeadKernel;      // nullable

    /**
     * Size of fused QKV output vector: qSize + 2 × kvSize.
     */
    public final int qkvFusedN;

    /**
     * Size of fused gate+up output vector: 2 × intermediateSize.
     */
    public final int gateUpFusedN;

    private boolean closed = false;

    private QwenGpuKernels(int gpuLayers, MatMulNBitsKernel[][] layerKernels,
                           MatMulNBitsKernel lmHeadKernel,
                           int qkvFusedN, int gateUpFusedN) {
        this.gpuLayers = gpuLayers;
        this.layerKernels = layerKernels;
        this.lmHeadKernel = lmHeadKernel;
        this.qkvFusedN = qkvFusedN;
        this.gateUpFusedN = gateUpFusedN;
    }

    // ── Factory ──────────────────────────────────────────────────────────

    /**
     * Create GPU kernels for the first {@code gpuLayers} decoder layers
     * and optionally the lm_head projection.
     *
     * @param wb        initialised WindowsBindings (D3D12 + DirectML)
     * @param weights   loaded Qwen2 weight set
     * @param config    model configuration
     * @param gpuLayers number of decoder layers to place on GPU (clipped to numHiddenLayers)
     * @param gpuLmHead whether to upload lm_head to GPU
     */
    public static QwenGpuKernels create(WindowsBindings wb, Qwen2Weights weights,
                                        Qwen2Config config, int gpuLayers,
                                        boolean gpuLmHead) {
        int layers = Math.min(Math.max(gpuLayers, 0), config.numHiddenLayers());
        int hidden = config.hiddenSize();
        int qSize = config.qSize();
        int kvSize = config.kvSize();
        int inter = config.intermediateSize();
        int qkvFusedN = qSize + 2 * kvSize;
        int gateUpFusedN = 2 * inter;

        log.info("Creating Qwen GPU kernels: {}/{} layers (QKV fused [{},{}], gate+up fused [{},{}]), lmHead={}",
                layers, config.numHiddenLayers(),
                qkvFusedN, hidden, gateUpFusedN, hidden, gpuLmHead);
        long t0 = System.currentTimeMillis();

        MatMulNBitsKernel[][] kernels = new MatMulNBitsKernel[layers][];
        for (int l = 0; l < layers; l++) {
            Qwen2Weights.LayerWeights lw = weights.layers[l];
            MatMulNBitsKernel[] k = new MatMulNBitsKernel[KERNELS_PER_LAYER];

            k[QKV_FUSED] = createFusedQKV(wb, lw, qSize, kvSize, hidden, l);
            k[O_PROJ] = createFromWeight(wb, lw.oProj(), "layer." + l + ".o_proj");
            k[GATE_UP_FUSED] = createFusedGateUp(wb, lw, inter, hidden, l);
            k[DOWN_PROJ] = createFromWeight(wb, lw.downProj(), "layer." + l + ".down_proj");
            kernels[l] = k;

            if ((l + 1) % 4 == 0 || l == layers - 1) {
                log.info("Qwen GPU kernels: {}/{} layers ({} ms)",
                        l + 1, layers, System.currentTimeMillis() - t0);
            }
        }

        MatMulNBitsKernel lmHead = null;
        if (gpuLmHead) {
            lmHead = createFromWeight(wb, weights.lmHead, "lm_head");
            log.info("Qwen GPU lm_head created [{}, {}]",
                    weights.lmHead.N(), weights.lmHead.K());
        }

        int total = layers * KERNELS_PER_LAYER + (lmHead != null ? 1 : 0);
        log.info("QwenGpuKernels ready: {} kernels ({} layers) in {} ms",
                total, layers, System.currentTimeMillis() - t0);

        return new QwenGpuKernels(layers, kernels, lmHead, qkvFusedN, gateUpFusedN);
    }

    // ── Kernel builders ──────────────────────────────────────────────────

    /**
     * Fused Q+K+V: dequantize/extract each projection separately,
     * concatenate rows into {@code [qSize+kvSize+kvSize, hidden]}, upload once.
     */
    private static MatMulNBitsKernel createFusedQKV(WindowsBindings wb,
                                                    Qwen2Weights.LayerWeights lw,
                                                    int qSize, int kvSize, int hidden,
                                                    int layerIdx) {
        float[] qDeq = dequantOrExtract(lw.qProj());
        float[] kDeq = dequantOrExtract(lw.kProj());
        float[] vDeq = dequantOrExtract(lw.vProj());

        int fusedN = qSize + 2 * kvSize;
        float[] fused = new float[fusedN * hidden];
        System.arraycopy(qDeq, 0, fused, 0, qSize * hidden);
        System.arraycopy(kDeq, 0, fused, qSize * hidden, kvSize * hidden);
        System.arraycopy(vDeq, 0, fused, (qSize + kvSize) * hidden, kvSize * hidden);

        log.debug("layer.{} QKV fused [{}, {}]", layerIdx, fusedN, hidden);
        return MatMulNBitsKernel.fromDequantizedWeights(wb, fusedN, hidden, fused);
    }

    /**
     * Fused gate+up: dequantize/extract each projection separately,
     * concatenate rows into {@code [2*intermediateSize, hidden]}, upload once.
     */
    private static MatMulNBitsKernel createFusedGateUp(WindowsBindings wb,
                                                       Qwen2Weights.LayerWeights lw,
                                                       int intermediate, int hidden,
                                                       int layerIdx) {
        float[] gDeq = dequantOrExtract(lw.gateProj());
        float[] uDeq = dequantOrExtract(lw.upProj());

        int fusedN = 2 * intermediate;
        float[] fused = new float[fusedN * hidden];
        System.arraycopy(gDeq, 0, fused, 0, intermediate * hidden);
        System.arraycopy(uDeq, 0, fused, intermediate * hidden, intermediate * hidden);

        log.debug("layer.{} gate+up fused [{}, {}]", layerIdx, fusedN, hidden);
        return MatMulNBitsKernel.fromDequantizedWeights(wb, fusedN, hidden, fused);
    }

    /**
     * Create a single GPU kernel from a {@link Qwen2Weights.WeightMatrix}.
     * Handles both quantized (INT4) and dense (FP32/FP16) weight types.
     */
    private static MatMulNBitsKernel createFromWeight(WindowsBindings wb,
                                                      Qwen2Weights.WeightMatrix wm,
                                                      String name) {
        if (wm instanceof Qwen2Weights.QuantizedWeightMatrix qwm) {
            Qwen2Weights.QuantizedWeight qw = qwm.inner();
            log.debug("GPU kernel (INT4): {} [{}, {}]", name, qw.N(), qw.K());
            return new MatMulNBitsKernel(wb, qw.N(), qw.K(),
                    qw.qWeight(), qw.scales(), qw.zeroPoints(), qw.blockSize());
        } else if (wm instanceof Qwen2Weights.DenseWeightMatrix dwm) {
            Qwen2Weights.DenseWeight dw = dwm.inner();
            log.debug("GPU kernel (FP32): {} [{}, {}]", name, dw.N(), dw.K());
            return MatMulNBitsKernel.fromDequantizedWeights(wb, dw.N(), dw.K(), dw.data());
        } else {
            throw new IllegalArgumentException(
                    "Unknown WeightMatrix type for kernel '" + name + "': " + wm.getClass());
        }
    }

    /**
     * Extract a flat FP32 weight array from a WeightMatrix.
     * For INT4: performs dequantization. For dense: returns data directly (no copy).
     */
    private static float[] dequantOrExtract(Qwen2Weights.WeightMatrix wm) {
        if (wm instanceof Qwen2Weights.QuantizedWeightMatrix qwm) {
            Qwen2Weights.QuantizedWeight qw = qwm.inner();
            return MatMulNBitsKernel.dequantizeInt4(
                    qw.qWeight(), qw.scales(), qw.zeroPoints(),
                    qw.N(), qw.K(), qw.blockSize());
        } else if (wm instanceof Qwen2Weights.DenseWeightMatrix dwm) {
            // Caller will copy into a fused buffer so aliasing is not an issue
            return dwm.inner().data();
        } else {
            throw new IllegalArgumentException(
                    "Unknown WeightMatrix type: " + wm.getClass());
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /**
     * True if layer {@code layerIdx} has GPU kernels.
     */
    public boolean hasLayer(int layerIdx) {
        return layerIdx >= 0 && layerIdx < gpuLayers;
    }

    /**
     * True if the lm_head projection has a GPU kernel.
     */
    public boolean hasLmHead() {
        return lmHeadKernel != null;
    }

    /**
     * Number of decoder layers with GPU kernels.
     */
    public int getGpuLayers() {
        return gpuLayers;
    }

    /**
     * Fused QKV kernel: output is a flat vector of length
     * {@link #qkvFusedN} = qSize + 2×kvSize.
     * Layout: Q[0..qSize), K[qSize..qSize+kvSize), V[qSize+kvSize..qkvFusedN).
     */
    public MatMulNBitsKernel qkvFused(int layer) {
        return layerKernels[layer][QKV_FUSED];
    }

    /**
     * o_proj kernel: maps attention output [qSize] → hidden state [hidden].
     */
    public MatMulNBitsKernel oProj(int layer) {
        return layerKernels[layer][O_PROJ];
    }

    /**
     * Fused gate+up kernel: output is a flat vector of length
     * {@link #gateUpFusedN} = 2×intermediateSize.
     * Layout: gate[0..intermediate), up[intermediate..gateUpFusedN).
     */
    public MatMulNBitsKernel gateUpFused(int layer) {
        return layerKernels[layer][GATE_UP_FUSED];
    }

    /**
     * down_proj kernel: maps MLP activation [intermediate] → hidden state [hidden].
     */
    public MatMulNBitsKernel downProj(int layer) {
        return layerKernels[layer][DOWN_PROJ];
    }

    /**
     * lm_head kernel (may be null if gpuLmHead=false).
     */
    public MatMulNBitsKernel lmHead() {
        return lmHeadKernel;
    }

    // ── AutoCloseable ────────────────────────────────────────────────────

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        int count = 0;
        for (MatMulNBitsKernel[] layer : layerKernels) {
            if (layer == null) continue;
            for (MatMulNBitsKernel k : layer) {
                if (k != null) {
                    k.close();
                    count++;
                }
            }
        }
        if (lmHeadKernel != null) {
            lmHeadKernel.close();
            count++;
        }
        log.info("QwenGpuKernels closed ({} kernels, {} layers)", count, gpuLayers);
    }
}
