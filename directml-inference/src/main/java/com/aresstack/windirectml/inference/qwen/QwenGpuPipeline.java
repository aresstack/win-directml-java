package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.windows.*;
import com.aresstack.windirectml.windows.QwenComputeShaders.ComputeKernelSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.MemorySegment;

/**
 * V2.0 batched GPU pipeline for Qwen2 decode.
 *
 * <p>Reduces fence waits per token by batching the MLP block
 * (o_proj + residual-add + RMSNorm + gate_up + SwiGLU + down_proj + residual-add)
 * into a single command list submission with one fence wait.
 *
 * <h2>Submission count per token (decode, 24 layers)</h2>
 * <pre>
 *   V1.0 (per-kernel matvec):  4 × 24 = 96  fence waits
 *   V2.0 (pipeline batch):     2 × 24 = 48  fence waits
 * </pre>
 *
 * <h2>Per-layer dispatch breakdown V2.0</h2>
 * <pre>
 *   Submission 1:  qkv_fused GEMM  → readback Q|K|V for CPU attention
 *   (CPU):         RoPE + KV-cache update + GQA attention → attnOutput
 *   Submission 2:  o_proj GEMM
 *                  + Add(oProj + hiddenInput → residual)
 *                  + RMSNorm(residual, postNormWeight → postNormed)
 *                  + gate_up_fused GEMM
 *                  + SwiGLU(gateUp → mlpAct)
 *                  + down_proj GEMM
 *                  + Add(residual + down → hiddenOut)
 *                  → readback hiddenOut
 * </pre>
 *
 * <h2>Differences from Phi3GpuPipeline</h2>
 * <ul>
 *   <li>No activation scale buffers ({@code mlpOutScale}) — Qwen2 does not use AWQ scales.</li>
 *   <li>SwiGLU uses {@link QwenComputeShaders#SWIGLU_HLSL} (no scale parameter).</li>
 *   <li>GQA: qkvFused output is {@code [qSize + 2*kvSize]} not {@code [3*hidden]}.</li>
 * </ul>
 */
public final class QwenGpuPipeline implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(QwenGpuPipeline.class);

    private final GpuPipeline pipeline;
    private final QwenGpuKernels kernels;
    private ComputeKernelSet computeKernels;   // null if shader compilation fails

    // ── GPU-resident intermediate buffer (residual, survives within a layer) ─
    private MemorySegment residualBuf;

    // ── Per-layer GPU-resident postNorm weight buffers (uploaded once) ───────
    private MemorySegment[] postNormWeightBufs;   // [numLayers]

    // ── Pre-allocated barriers ──────────────────────────────────────────────
    private MemorySegment uavBarrier;
    private MemorySegment barrierResidualCopyDestToUav;
    private MemorySegment barrierResidualUavToCopySource;

    private final int hidden;
    private final int intermediate;
    private final float rmsNormEps;
    private boolean mlpBatchEnabled = false;
    private boolean weightsUploaded = false;
    private boolean closed = false;

    // ── Constructor ──────────────────────────────────────────────────────────

    /**
     * Create the V2.0 batched GPU pipeline for Qwen2 decode.
     *
     * @param wb      initialised WindowsBindings
     * @param kernels GPU kernel set (weights already uploaded)
     * @param config  Qwen2 model configuration
     * @throws WindowsNativeException if pipeline creation fails
     */
    public QwenGpuPipeline(WindowsBindings wb, QwenGpuKernels kernels, Qwen2Config config)
            throws WindowsNativeException {
        this.kernels = kernels;
        this.hidden = config.hiddenSize();
        this.intermediate = config.intermediateSize();
        this.rmsNormEps = config.rmsNormEps();

        // Size the shared staging buffers:
        //   upload   = max(hidden, qkvFusedN)  — largest GEMM input
        //   readback = max(qkvFusedN, hidden)  — largest GEMM output + vocab for lm_head
        long qkvBytes = (long) kernels.qkvFusedN * Float.BYTES;
        long hiddenBytes = (long) hidden * Float.BYTES;
        long vocabBytes = (long) config.vocabSize() * Float.BYTES;
        long maxUpload = hiddenBytes;
        long maxReadback = Math.max(qkvBytes, Math.max(vocabBytes, hiddenBytes));

        this.pipeline = new GpuPipeline(wb, maxUpload, maxReadback);

        // ── Compile V2.0 compute shaders ─────────────────────────────────────
        try {
            computeKernels = QwenComputeShaders.createAll(wb, pipeline.getCommandList());
            mlpBatchEnabled = true;
            log.info("QwenGpuPipeline V2.0: MLP batch mode (48 submissions/token for 24 layers)");
        } catch (Exception e) {
            log.warn("V2.0 shader compilation failed – falling back to per-kernel dispatch: {}",
                    e.getMessage());
            computeKernels = null;
            mlpBatchEnabled = false;
        }

        // ── MLP batch buffers (only if shaders compiled) ─────────────────────
        if (mlpBatchEnabled) {
            var dev = wb.getD3d12Device();
            var arena = pipeline.getArena();

            residualBuf = D3D12Bindings.createDefaultBuffer(dev, hiddenBytes, arena);

            uavBarrier = pipeline.allocUavBarrier();
            barrierResidualCopyDestToUav = pipeline.allocTransitionBarrier(residualBuf,
                    D3D12Bindings.D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            barrierResidualUavToCopySource = pipeline.allocTransitionBarrier(residualBuf,
                    D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12Bindings.D3D12_RESOURCE_STATE_COPY_SOURCE);
        }

        log.info("QwenGpuPipeline ready: mlpBatch={}, upload={}KB, readback={}KB",
                mlpBatchEnabled, maxUpload / 1024, maxReadback / 1024);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Single-GEMM dispatch (shared pipeline — avoids per-kernel CL overhead)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Execute a single GEMM via the shared pipeline.
     * Faster than {@code MatMulNBitsKernel.matvec()} because the command allocator,
     * fence, and staging buffers are shared across all calls.
     */
    public void matvec(MatMulNBitsKernel kernel, float[] input, float[] output) {
        pipeline.begin();
        kernel.recordInto(pipeline, input);
        pipeline.submitAndWait();
        kernel.readResult(output);
    }

    /**
     * Execute the fused QKV projection via the shared pipeline.
     * Output layout: {@code [Q | K | V]}, split by the caller.
     */
    public void qkvFused(int layerIdx, float[] normedHidden, float[] decQKV) {
        matvec(kernels.qkvFused(layerIdx), normedHidden, decQKV);
    }

    /**
     * Execute the lm_head projection via the shared pipeline.
     */
    public void lmHead(float[] hidden, float[] logits) {
        matvec(kernels.lmHead(), hidden, logits);
    }

    /**
     * Execute the o_proj via the shared pipeline (fallback if MLP batch not available).
     */
    public void oProj(int layerIdx, float[] attnOut, float[] out) {
        matvec(kernels.oProj(layerIdx), attnOut, out);
    }

    /**
     * Execute the fused gate+up via the shared pipeline (fallback if MLP batch not available).
     */
    public void gateUpFused(int layerIdx, float[] normed, float[] out) {
        matvec(kernels.gateUpFused(layerIdx), normed, out);
    }

    /**
     * Execute the down_proj via the shared pipeline (fallback if MLP batch not available).
     */
    public void downProj(int layerIdx, float[] mlpAct, float[] out) {
        matvec(kernels.downProj(layerIdx), mlpAct, out);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Batched GEMM helpers (Opt-A — prefill submission collapse)
    //
    // Each call submits ONE GPU dispatch over M rows instead of M dispatches
    // of one row each. Used by Qwen2Runtime.processLayerPrefill to turn the
    // per-token-row loop into a single GEMM per projection per layer.
    // Decode keeps using the per-row qkvFused / oProj / gateUpFused / downProj.
    // ══════════════════════════════════════════════════════════════════════

    /**
     * True if all four projection kernels (qkvFused, oProj, gateUpFused, downProj)
     * support the batched matmul path. False if any of them is on the legacy
     * FP32/DML fallback that does not support batching.
     */
    public boolean supportsBatch(int layerIdx) {
        return kernels.qkvFused(layerIdx).supportsBatch()
                && kernels.oProj(layerIdx).supportsBatch()
                && kernels.gateUpFused(layerIdx).supportsBatch()
                && kernels.downProj(layerIdx).supportsBatch();
    }

    /**
     * Batched fused QKV: {@code [M, hidden] → [M, qSize + 2*kvSize]}.
     */
    public void qkvFusedBatch(int layerIdx, float[] normedBatch, float[] qkvBatch, int M) {
        kernels.qkvFused(layerIdx).matmulBatch(normedBatch, qkvBatch, M);
    }

    /**
     * Batched o_proj: {@code [M, qSize] → [M, hidden]}.
     */
    public void oProjBatch(int layerIdx, float[] attnBatch, float[] outBatch, int M) {
        kernels.oProj(layerIdx).matmulBatch(attnBatch, outBatch, M);
    }

    /**
     * Batched fused gate+up: {@code [M, hidden] → [M, 2*intermediate]}.
     */
    public void gateUpFusedBatch(int layerIdx, float[] normedBatch, float[] gateUpBatch, int M) {
        kernels.gateUpFused(layerIdx).matmulBatch(normedBatch, gateUpBatch, M);
    }

    /**
     * Batched down_proj: {@code [M, intermediate] → [M, hidden]}.
     */
    public void downProjBatch(int layerIdx, float[] mlpActBatch, float[] downBatch, int M) {
        kernels.downProj(layerIdx).matmulBatch(mlpActBatch, downBatch, M);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MLP Batch: 6 GPU ops, 1 submission (V2.0)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Whether MLP batching is available (shaders compiled + postNorm weights uploaded).
     */
    public boolean isMlpBatchEnabled() {
        return mlpBatchEnabled && weightsUploaded;
    }

    /**
     * Upload per-layer postNorm weights to GPU so RMSNorm can run GPU-resident.
     * Must be called once after construction, before the first {@link #batchMlp} call.
     *
     * @param wb      initialised WindowsBindings
     * @param weights loaded Qwen2 weight set
     * @param config  model configuration
     * @throws WindowsNativeException if buffer creation or upload fails
     */
    public void uploadLayerWeights(WindowsBindings wb, Qwen2Weights weights, Qwen2Config config)
            throws WindowsNativeException {
        if (!mlpBatchEnabled) return;

        var dev = wb.getD3d12Device();
        var queue = wb.getCommandQueue();
        var arena = pipeline.getArena();
        int nLayers = config.numHiddenLayers();
        long hiddenBytes = (long) hidden * Float.BYTES;

        postNormWeightBufs = new MemorySegment[nLayers];

        long t0 = System.currentTimeMillis();
        for (int l = 0; l < nLayers; l++) {
            float[] w = weights.layers[l].postNormWeight();
            postNormWeightBufs[l] = D3D12Bindings.createDefaultBuffer(dev, hiddenBytes, arena);
            D3D12Bindings.uploadFloats(dev, queue, postNormWeightBufs[l], w, arena);
        }

        weightsUploaded = true;
        log.info("Qwen postNorm weights uploaded to GPU: {} layers in {} ms",
                nLayers, System.currentTimeMillis() - t0);
    }

    /**
     * Batched MLP: 6 GPU operations in ONE submission.
     *
     * <p>Operations (all GPU-resident, no intermediate CPU round-trips):
     * <ol>
     *   <li>Upload {@code attnOutput} → GPU, dispatch o_proj GEMM</li>
     *   <li>Upload {@code hiddenInput} → {@code residualBuf}; Add(oProjOut + residual → residual)</li>
     *   <li>RMSNorm(residual, postNormWeight[layer] → gateUpFused.inputBuf)</li>
     *   <li>gate_up_fused GEMM dispatch</li>
     *   <li>SwiGLU(gateUpOut → downProj.inputBuf)</li>
     *   <li>down_proj GEMM; Add(residual + downOut → residual); readback</li>
     * </ol>
     *
     * @param attnOutput  attention output [qSize] — becomes o_proj input
     * @param hiddenInput hidden state entering this layer [hidden] — residual source
     * @param hiddenOut   where to write the layer output [hidden] — may alias hiddenInput
     * @param layerIdx    decoder layer index
     */
    public void batchMlp(float[] attnOutput, float[] hiddenInput, float[] hiddenOut,
                         int layerIdx) {
        if (!isMlpBatchEnabled()) {
            throw new IllegalStateException(
                    "MLP batch not enabled — shaders failed or weights not uploaded");
        }

        long hiddenBytes = (long) hidden * Float.BYTES;

        MatMulNBitsKernel oK = kernels.oProj(layerIdx);
        MatMulNBitsKernel guK = kernels.gateUpFused(layerIdx);
        MatMulNBitsKernel downK = kernels.downProj(layerIdx);

        pipeline.begin();
        var cl = pipeline.getCommandList();

        // 1. o_proj: upload attnOutput, dispatch → oProjOut (GPU-resident)
        oK.recordBatchFromCpu(pipeline, attnOutput);

        // 2. Upload hiddenInput → residualBuf  (will become residual1 = oProj + hidden)
        pipeline.recordUpload(hiddenInput, 0, hidden, residualBuf, 0);
        pipeline.recordBarrier(barrierResidualCopyDestToUav);
        pipeline.recordUavBarrier(uavBarrier);

        // 3. residual1 = oProjOut + hiddenInput  (in-place: residualBuf ← residualBuf + oProjOut)
        computeKernels.add().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(oK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf)
                },
                new int[]{hidden},
                hidden);
        pipeline.recordUavBarrier(uavBarrier);

        // 4. RMSNorm(residual1, postNormWeight → gateUpFused.inputBuf)
        int epsBits = Float.floatToRawIntBits(rmsNormEps);
        computeKernels.rmsNorm().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(postNormWeightBufs[layerIdx]),
                        D3D12Bindings.getGpuVirtualAddress(guK.getInputBuf())
                },
                new int[]{hidden, epsBits},
                1);
        pipeline.recordUavBarrier(uavBarrier);

        // 5. gate_up_fused GEMM (input already in GPU inputBuf from RMSNorm output)
        guK.recordBatchDispatchOnly(pipeline);
        pipeline.recordUavBarrier(uavBarrier);

        // 6. SwiGLU: silu(gate) * up  — no activation scale for Qwen
        computeKernels.swiglu().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(guK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(downK.getInputBuf())
                },
                new int[]{intermediate},
                intermediate);
        pipeline.recordUavBarrier(uavBarrier);

        // 7. down_proj GEMM (input in downK.inputBuf from SwiGLU output)
        downK.recordBatchDispatchOnly(pipeline);
        pipeline.recordUavBarrier(uavBarrier);

        // 8. residual2 = residual1 + downOut  (in-place: residualBuf ← residualBuf + downOut)
        computeKernels.add().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(downK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf)
                },
                new int[]{hidden},
                hidden);

        // 9. Transition residualBuf → COPY_SOURCE, then readback
        pipeline.recordBarrier(barrierResidualUavToCopySource);
        pipeline.recordReadback(residualBuf, 0, hiddenBytes);
        pipeline.submitAndWait();

        // 10. Copy result back to CPU
        pipeline.readbackInto(hiddenOut, 0, hidden);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Accessors
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * True if layer {@code layerIdx} has GPU kernels.
     */
    public boolean hasLayer(int layerIdx) {
        return kernels.hasLayer(layerIdx);
    }

    /**
     * True if the lm_head projection has a GPU kernel.
     */
    public boolean hasLmHead() {
        return kernels.hasLmHead();
    }

    /**
     * The underlying shared pipeline (for advanced use).
     */
    public GpuPipeline getPipeline() {
        return pipeline;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // AutoCloseable
    // ═══════════════════════════════════════════════════════════════════════════

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (computeKernels != null) computeKernels.close();
        pipeline.close();
        log.info("QwenGpuPipeline closed");
    }
}
