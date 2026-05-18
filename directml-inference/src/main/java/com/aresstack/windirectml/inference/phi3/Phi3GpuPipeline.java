package com.aresstack.windirectml.inference.phi3;

import com.aresstack.windirectml.windows.*;
import com.aresstack.windirectml.windows.Phi3ComputeShaders.ComputeKernelSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.MemorySegment;

/**
 * V2.0 batched GPU pipeline for Phi-3 decode.
 * <p>
 * Collapses 129 → 65 submissions per token by batching the MLP block
 * (O-proj + residual-add + RMSNorm + GateUp + SwiGLU + Down + residual-add)
 * into a single command list submission with one fence wait.
 * <p>
 * <b>Submission count</b>: 129 (V1.x per-kernel) → <b>65</b> (V2.0 MLP batch)
 */
public final class Phi3GpuPipeline implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(Phi3GpuPipeline.class);

    private final GpuPipeline pipeline;
    private final Phi3GpuKernels kernels;
    private ComputeKernelSet computeKernels;  // nullable if shader compilation fails

    // ── GPU-resident intermediate buffer ─────────────────────────────
    private MemorySegment residualBuf;

    // ── GPU-resident weight buffers (per-layer, uploaded once) ───────
    private MemorySegment[] postNormWeightBufs;
    private MemorySegment[] mlpOutScaleBufs;

    // ── Pre-allocated barriers ──────────────────────────────────────
    private MemorySegment uavBarrier;
    private MemorySegment barrierResidualCopyDestToUav;
    private MemorySegment barrierResidualUavToCopySource;

    private final int hidden;
    private final int intermediate;
    private final float rmsNormEps;
    private boolean mlpBatchEnabled = false;
    private boolean weightsUploaded = false;
    private boolean closed = false;

    /**
     * Create the V2.0 batched GPU pipeline.
     */
    public Phi3GpuPipeline(WindowsBindings wb, Phi3GpuKernels kernels, Phi3Config config)
            throws WindowsNativeException {
        this.kernels = kernels;
        this.hidden = config.hiddenSize();
        this.intermediate = config.intermediateSize();
        this.rmsNormEps = config.rmsNormEps();

        long hiddenBytes = (long) hidden * Float.BYTES;
        long qkvBytes = (long) hidden * 3 * Float.BYTES;
        long vocabBytes = (long) config.vocabSize() * Float.BYTES;

        long maxUpload = hiddenBytes;
        long maxReadback = Math.max(qkvBytes, Math.max(vocabBytes, hiddenBytes));

        this.pipeline = new GpuPipeline(wb, maxUpload, maxReadback);

        // ── Compile V2.0 compute shaders ─────────────────────────────
        try {
            computeKernels = Phi3ComputeShaders.createAll(wb, pipeline.getCommandList());
            mlpBatchEnabled = true;
            log.info("Phi3GpuPipeline V2.0: MLP batch mode (65 submissions/token)");
        } catch (Exception e) {
            log.warn("V2.0 shader compilation failed: {}", e.getMessage());
            computeKernels = null;
            mlpBatchEnabled = false;
        }

        // ── MLP batch buffers ────────────────────────────────────────
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

        log.info("Phi3GpuPipeline ready: mlpBatch={}, upload={}KB readback={}KB",
                mlpBatchEnabled, maxUpload / 1024, maxReadback / 1024);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Single-GEMM dispatch using shared pipeline
    // ═══════════════════════════════════════════════════════════════════

    /** Execute a single GEMM via the shared pipeline. */
    public void matvec(MatMulNBitsKernel kernel, float[] input, float[] output) {
        pipeline.begin();
        kernel.recordInto(pipeline, input);
        pipeline.submitAndWait();
        kernel.readResult(output);
    }

    public void qkvFused(int layerIdx, float[] input, float[] qkvOutput) {
        matvec(kernels.qkvFused(layerIdx), input, qkvOutput);
    }

    public void oProj(int layerIdx, float[] input, float[] output) {
        matvec(kernels.oProj(layerIdx), input, output);
    }

    public void gateUpProj(int layerIdx, float[] input, float[] output) {
        matvec(kernels.gateUpProj(layerIdx), input, output);
    }

    public void downProj(int layerIdx, float[] input, float[] output) {
        matvec(kernels.downProj(layerIdx), input, output);
    }

    public void lmHead(float[] input, float[] logits) {
        matvec(kernels.lmHead(), input, logits);
    }

    // ═══════════════════════════════════════════════════════════════════
    // MLP Batch: 7 GPU ops, 1 submission (V2.0)
    // ═══════════════════════════════════════════════════════════════════

    /** Whether MLP batching is available (compute shaders compiled + weights uploaded). */
    public boolean isMlpBatchEnabled() { return mlpBatchEnabled && weightsUploaded; }

    /**
     * Upload per-layer weights to GPU (for MLP batch).
     */
    public void uploadLayerWeights(WindowsBindings wb, Phi3Weights weights, Phi3Config config)
            throws WindowsNativeException {
        if (!mlpBatchEnabled) return;

        var dev = wb.getD3d12Device();
        var queue = wb.getCommandQueue();
        var arena = pipeline.getArena();
        int nLayers = config.numHiddenLayers();
        long hiddenBytes = (long) hidden * Float.BYTES;
        long interBytes = (long) intermediate * Float.BYTES;

        postNormWeightBufs = new MemorySegment[nLayers];
        mlpOutScaleBufs = new MemorySegment[nLayers];

        long t0 = System.currentTimeMillis();
        for (int l = 0; l < nLayers; l++) {
            var lw = weights.layers[l];
            postNormWeightBufs[l] = D3D12Bindings.createDefaultBuffer(dev, hiddenBytes, arena);
            D3D12Bindings.uploadFloats(dev, queue, postNormWeightBufs[l], lw.postNormWeight(), arena);
            mlpOutScaleBufs[l] = D3D12Bindings.createDefaultBuffer(dev, interBytes, arena);
            D3D12Bindings.uploadFloats(dev, queue, mlpOutScaleBufs[l], lw.mlpOutScale(), arena);
        }

        weightsUploaded = true;
        long elapsed = System.currentTimeMillis() - t0;
        log.info("Uploaded {} weight buffers to GPU in {} ms", nLayers * 2, elapsed);
    }

    /**
     * Batched MLP: 7 GPU operations in ONE submission.
     * <p>
     * Operations: O-proj GEMM → residual-add → RMSNorm → GateUp GEMM →
     * SwiGLU+Scale → Down GEMM → residual-add → readback.
     */
    public void batchMlp(float[] attnOutput, float[] hiddenInput, float[] hiddenOut,
                          int layerIdx) {
        if (!isMlpBatchEnabled()) {
            throw new IllegalStateException("MLP batch not enabled");
        }

        long hiddenBytes = (long) hidden * Float.BYTES;

        MatMulNBitsKernel oK = kernels.oProj(layerIdx);
        MatMulNBitsKernel guK = kernels.gateUpProj(layerIdx);
        MatMulNBitsKernel downK = kernels.downProj(layerIdx);

        pipeline.begin();
        var cl = pipeline.getCommandList();

        oK.recordBatchFromCpu(pipeline, attnOutput);
        pipeline.recordUpload(hiddenInput, 0, hidden, residualBuf, 0);
        pipeline.recordBarrier(barrierResidualCopyDestToUav);
        pipeline.recordUavBarrier(uavBarrier);

        computeKernels.add().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(oK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf)
                },
                new int[]{ hidden },
                hidden);

        pipeline.recordUavBarrier(uavBarrier);

        int epsBits = Float.floatToRawIntBits(rmsNormEps);
        computeKernels.rmsNorm().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(postNormWeightBufs[layerIdx]),
                        D3D12Bindings.getGpuVirtualAddress(guK.getInputBuf())
                },
                new int[]{ hidden, epsBits },
                1);

        pipeline.recordUavBarrier(uavBarrier);
        guK.recordBatchDispatchOnly(pipeline);
        pipeline.recordUavBarrier(uavBarrier);

        computeKernels.swiglu().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(guK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(mlpOutScaleBufs[layerIdx]),
                        D3D12Bindings.getGpuVirtualAddress(downK.getInputBuf())
                },
                new int[]{ intermediate },
                intermediate);

        pipeline.recordUavBarrier(uavBarrier);
        downK.recordBatchDispatchOnly(pipeline);
        pipeline.recordUavBarrier(uavBarrier);

        computeKernels.add().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(downK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf)
                },
                new int[]{ hidden },
                hidden);

        pipeline.recordBarrier(barrierResidualUavToCopySource);
        pipeline.recordReadback(residualBuf, 0, hiddenBytes);
        pipeline.submitAndWait();
        pipeline.readbackInto(hiddenOut, 0, hidden);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Accessors
    // ═══════════════════════════════════════════════════════════════════

    public boolean hasLayer(int layerIdx) { return kernels.hasLayer(layerIdx); }
    public boolean hasLmHead() { return kernels.hasLmHead(); }
    public GpuPipeline getPipeline() { return pipeline; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (computeKernels != null) computeKernels.close();
        pipeline.close();
        log.info("Phi3GpuPipeline closed");
    }
}
