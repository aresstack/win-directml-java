package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.windows.*;
import com.aresstack.windirectml.windows.QwenComputeShaders.ComputeKernelSet;
import com.aresstack.windirectml.windows.QwenAttentionShaders.AttentionKernelSet;
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
    private AttentionKernelSet attentionKernels;  // null if Opt-B shaders fail to compile
    private QwenGpuKvCache kvCache;             // null until setKvCache() is called

    // ── GPU-resident intermediate buffer (residual, survives within a layer) ─
    private MemorySegment residualBuf;

    // ── Per-layer GPU-resident postNorm weight buffers (uploaded once) ───────
    private MemorySegment[] postNormWeightBufs;   // [numLayers]

    // ── Per-layer fused QKV bias buffers [qSize + 2*kvSize] (Opt-B; null when no biases) ──
    private MemorySegment[] qkvBiasBufs;          // [numLayers] entry may be null
    private boolean anyLayerHasQkvBias = false;

    // ── Pre-allocated barriers ──────────────────────────────────────────────
    private MemorySegment uavBarrier;
    private MemorySegment barrierResidualCopyDestToUav;
    private MemorySegment barrierResidualUavToCopySource;

    private final int hidden;
    private final int intermediate;
    private final float rmsNormEps;
    // Cached for Opt-B GPU-resident attention path
    private final int numHeads;
    private final int kvHeads;
    private final int headDim;
    private final int qSize;
    private final int kvSize;
    private final int qHeadsPerKvHead;
    private final float ropeTheta;
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
        this.numHeads = config.numAttentionHeads();
        this.kvHeads = config.numKeyValueHeads();
        this.headDim = config.headDim();
        this.qSize = config.qSize();
        this.kvSize = config.kvSize();
        this.qHeadsPerKvHead = numHeads / kvHeads;
        this.ropeTheta = config.ropeTheta();

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

        // Compile Opt-B attention shaders (rope_and_append + gqa_attention_decode)
        try {
            attentionKernels = QwenAttentionShaders.createAll(wb, pipeline.getCommandList());
            log.info("QwenGpuPipeline Opt-B: attention shaders compiled (GPU-resident decode available once setKvCache is called)");
        } catch (Exception e) {
            log.warn("Opt-B attention shader compilation failed - GPU-resident decode disabled: {}",
                    e.getMessage());
            attentionKernels = null;
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

        // Upload per-layer fused QKV biases (Opt-B requires them on GPU for
        // GPU-resident decode; Qwen2 always has q/k/v biases, but we tolerate
        // any subset being null and zero-fill the missing slices).
        qkvBiasBufs = new MemorySegment[nLayers];
        long qkvBiasFloats = (long) qSize + 2L * kvSize;
        long qkvBiasBytes = qkvBiasFloats * Float.BYTES;
        int uploadedBiasLayers = 0;
        for (int l = 0; l < nLayers; l++) {
            var lw = weights.layers[l];
            float[] qB = lw.qBias();
            float[] kB = lw.kBias();
            float[] vB = lw.vBias();
            if (qB == null && kB == null && vB == null) {
                qkvBiasBufs[l] = null;
                continue;
            }
            float[] fused = new float[(int) qkvBiasFloats];
            if (qB != null) System.arraycopy(qB, 0, fused, 0, qSize);
            if (kB != null) System.arraycopy(kB, 0, fused, qSize, kvSize);
            if (vB != null) System.arraycopy(vB, 0, fused, qSize + kvSize, kvSize);
            qkvBiasBufs[l] = D3D12Bindings.createDefaultBuffer(dev, qkvBiasBytes, arena);
            D3D12Bindings.uploadFloats(dev, queue, qkvBiasBufs[l], fused, arena);
            uploadedBiasLayers++;
        }
        anyLayerHasQkvBias = uploadedBiasLayers > 0;

        weightsUploaded = true;
        log.info("Qwen postNorm weights uploaded to GPU: {} layers, qkvBias layers={}/{} ({} KB each) in {} ms",
                nLayers, uploadedBiasLayers, nLayers, qkvBiasBytes / 1024,
                System.currentTimeMillis() - t0);
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
    // Opt-B: GPU-resident decode (QKV + RoPE + GQA attention + MLP in ONE submission)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Attach the GPU-resident KV cache. Must be called once after construction
     * (and after {@link #uploadLayerWeights}) before the first
     * {@link #decodeLayerGpuResident} call.
     *
     * <p>Passing a cache whose dimensions disagree with the model config is a bug
     * and is rejected.
     */
    public void setKvCache(QwenGpuKvCache cache) {
        if (cache.getKvHeads() != kvHeads || cache.getHeadDim() != headDim) {
            throw new IllegalArgumentException("KV cache shape mismatch: cache="
                    + cache.getKvHeads() + "x" + cache.getHeadDim()
                    + " config=" + kvHeads + "x" + headDim);
        }
        this.kvCache = cache;
        log.info("QwenGpuPipeline Opt-B: KV cache attached (maxSeqLen={}, {} layers) - GPU-resident decode enabled={}",
                cache.getMaxSeqLen(), cache.getNumLayers(), isAttnGpuResidentEnabled());
    }

    /**
     * True iff the full Opt-B GPU-resident decode path is available:
     * MLP-batch shaders compiled, postNorm weights uploaded, attention shaders
     * compiled, and a KV cache attached.
     */
    public boolean isAttnGpuResidentEnabled() {
        return mlpBatchEnabled && weightsUploaded
                && attentionKernels != null && kvCache != null;
    }

    /**
     * Upload prefill CPU-built KV cache for one layer into the GPU-resident
     * cache (B-Step 6). Delegates to {@link QwenGpuKvCache#uploadFromCpu}.
     *
     * @param layerIdx layer index
     * @param seqLen   number of valid positions (prompt length)
     * @param cpuK     CPU K cache: {@code float[kvHeads][>= seqLen*headDim]}
     * @param cpuV     CPU V cache: {@code float[kvHeads][>= seqLen*headDim]}
     * @throws IllegalStateException if no KV cache is attached
     */
    public void uploadKvCacheFromCpu(int layerIdx, int seqLen,
                                     float[][] cpuK, float[][] cpuV) {
        if (kvCache == null) {
            throw new IllegalStateException(
                    "Cannot upload KV cache - no QwenGpuKvCache attached (call setKvCache first)");
        }
        kvCache.uploadFromCpu(layerIdx, seqLen, cpuK, cpuV);
    }

    /**
     * Single-submission GPU-resident decoder layer (B-Step 3+4 fused with C-Step 1).
     *
     * <p>Records 9 GPU operations into ONE command list and fence-waits once:
     * <ol>
     *   <li>Upload {@code normedHidden} -> qkv input; qkv_fused GEMM</li>
     *   <li>rope_and_append: rotate Q in place; rotate K and append K|V to KV cache[pos]</li>
     *   <li>gqa_attention_decode: read Q + KV cache[0..pos] -> write attnOut into oProj.input</li>
     *   <li>o_proj GEMM (input already GPU-resident)</li>
     *   <li>Upload {@code hiddenInput} -> residualBuf; Add(oProj + residual -> residual)</li>
     *   <li>RMSNorm(residual, postNormWeight -> gateUp.input)</li>
     *   <li>gate_up_fused GEMM; SwiGLU -> down.input</li>
     *   <li>down_proj GEMM; Add(residual + down -> residual)</li>
     *   <li>Transition residual -> COPY_SOURCE; readback into {@code hiddenOut}</li>
     * </ol>
     *
     * <p>Per token (24 layers): 24 submissions vs. 48 with V2.0 MLP-batch only.
     * Q/K/V/attnOut never cross the CPU boundary; KV cache lives entirely on the GPU.
     *
     * @param layerIdx     decoder layer index
     * @param normedHidden RMSNorm(hiddenInput) [hidden] - input to QKV
     * @param hiddenInput  hidden state entering this layer [hidden] - residual source
     * @param hiddenOut    where to write the layer output [hidden] - may alias hiddenInput
     * @param pos          current decode position (the slot KV will be written to)
     */
    public void decodeLayerGpuResident(int layerIdx, float[] normedHidden,
                                       float[] hiddenInput, float[] hiddenOut,
                                       int pos) {
        if (!isAttnGpuResidentEnabled()) {
            throw new IllegalStateException(
                    "Opt-B GPU-resident decode not available: mlpBatch=" + mlpBatchEnabled
                            + " weightsUploaded=" + weightsUploaded
                            + " attentionKernels=" + (attentionKernels != null)
                            + " kvCache=" + (kvCache != null));
        }
        if (pos < 0 || pos >= kvCache.getMaxSeqLen()) {
            throw new IllegalArgumentException("pos=" + pos
                    + " out of [0.." + kvCache.getMaxSeqLen() + ")");
        }

        long hiddenBytes = (long) hidden * Float.BYTES;

        MatMulNBitsKernel qkvK = kernels.qkvFused(layerIdx);
        MatMulNBitsKernel oK = kernels.oProj(layerIdx);
        MatMulNBitsKernel guK = kernels.gateUpFused(layerIdx);
        MatMulNBitsKernel downK = kernels.downProj(layerIdx);

        pipeline.begin();
        var cl = pipeline.getCommandList();

        // ── 1. QKV GEMM: upload normedHidden + dispatch → qkvK.outputBuf = [Q|K|V] ──
        qkvK.recordBatchFromCpu(pipeline, normedHidden);
        pipeline.recordUavBarrier(uavBarrier);

        // ── 1b. Add per-layer fused QKV bias (Qwen2 has q/k/v biases) ──
        // No-op if layer has no biases.
        long qkvBaseAddr = D3D12Bindings.getGpuVirtualAddress(qkvK.getOutputBuf());
        int qkvN = qSize + 2 * kvSize;
        if (qkvBiasBufs != null && qkvBiasBufs[layerIdx] != null) {
            computeKernels.add().recordDispatch(cl,
                    new long[]{
                            qkvBaseAddr,
                            D3D12Bindings.getGpuVirtualAddress(qkvBiasBufs[layerIdx]),
                            qkvBaseAddr
                    },
                    new int[]{qkvN},
                    qkvN);
            pipeline.recordUavBarrier(uavBarrier);
        }

        // ── 2. ROPE + APPEND KV: bind one fused buffer to (Q,K,V) UAVs via byte offsets ──
        // After: Q rotated in place at qkvBase..qkvBase+qSize;
        //        rotated K and V written to KV cache[layer][..][pos]
        long qkvBase = qkvBaseAddr;
        long qAddr = qkvBase;
        long kAddr = qkvBase + (long) qSize * Float.BYTES;
        long vAddr = qkvBase + (long) (qSize + kvSize) * Float.BYTES;
        long kCacheAddr = kvCache.getKAddr(layerIdx);
        long vCacheAddr = kvCache.getVAddr(layerIdx);
        int ropeThetaBits = Float.floatToRawIntBits(ropeTheta);
        int halfDim = headDim / 2;
        attentionKernels.ropeAndAppend().recordDispatch(cl,
                new long[]{qAddr, kAddr, vAddr, kCacheAddr, vCacheAddr},
                new int[]{numHeads, kvHeads, headDim, pos,
                        kvCache.getMaxSeqLen(), ropeThetaBits},
                (numHeads + kvHeads) * halfDim);
        pipeline.recordUavBarrier(uavBarrier);

        // ── 3. GQA attention decode: Q + KCache[0..pos] + VCache[0..pos] → oK.inputBuf ──
        long attnOutAddr = D3D12Bindings.getGpuVirtualAddress(oK.getInputBuf());
        float scale = (float) (1.0 / Math.sqrt(headDim));
        int scaleBits = Float.floatToRawIntBits(scale);
        int seqLen = pos + 1;
        attentionKernels.gqaAttentionDecode().recordDispatch(cl,
                new long[]{qAddr, kCacheAddr, vCacheAddr, attnOutAddr},
                new int[]{numHeads, kvHeads, headDim, qHeadsPerKvHead,
                        seqLen, kvCache.getMaxSeqLen(), scaleBits},
                numHeads * headDim);
        pipeline.recordUavBarrier(uavBarrier);

        // ── 4. o_proj GEMM (input is already in oK.inputBuf from GQA attention) ──
        oK.recordBatchDispatchOnly(pipeline);

        // ── 5. Upload hiddenInput → residualBuf; Add(oProj + residual → residual) ──
        pipeline.recordUpload(hiddenInput, 0, hidden, residualBuf, 0);
        pipeline.recordBarrier(barrierResidualCopyDestToUav);
        pipeline.recordUavBarrier(uavBarrier);

        computeKernels.add().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(oK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf)
                },
                new int[]{hidden},
                hidden);
        pipeline.recordUavBarrier(uavBarrier);

        // ── 6. RMSNorm(residual, postNormWeight → gateUp.input) ──
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

        // ── 7. gate_up GEMM, SwiGLU, down_proj GEMM ──
        guK.recordBatchDispatchOnly(pipeline);
        pipeline.recordUavBarrier(uavBarrier);

        computeKernels.swiglu().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(guK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(downK.getInputBuf())
                },
                new int[]{intermediate},
                intermediate);
        pipeline.recordUavBarrier(uavBarrier);

        downK.recordBatchDispatchOnly(pipeline);
        pipeline.recordUavBarrier(uavBarrier);

        // ── 8. Add(residual + downOut → residual) ──
        computeKernels.add().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(downK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf)
                },
                new int[]{hidden},
                hidden);

        // ── 9. Transition residualBuf → COPY_SOURCE, readback into hiddenOut ──
        pipeline.recordBarrier(barrierResidualUavToCopySource);
        pipeline.recordReadback(residualBuf, 0, hiddenBytes);
        pipeline.submitAndWait();
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
        if (attentionKernels != null) attentionKernels.close();
        if (computeKernels != null) computeKernels.close();
        pipeline.close();
        log.info("QwenGpuPipeline closed");
    }
}
