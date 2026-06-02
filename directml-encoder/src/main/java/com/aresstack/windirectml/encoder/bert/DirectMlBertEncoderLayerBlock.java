package com.aresstack.windirectml.encoder.bert;

import com.aresstack.windirectml.runtime.DirectMlContextImpl;
import com.aresstack.windirectml.runtime.DirectMlGpuBatch;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;
import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.GpuBuffer;
import com.aresstack.windirectml.runtime.TensorDataType;
import com.aresstack.windirectml.runtime.TensorLayout;
import com.aresstack.windirectml.runtime.TensorShape;
import com.aresstack.windirectml.runtime.kernels.DirectMlAddKernel;
import com.aresstack.windirectml.runtime.kernels.DirectMlAttentionKernel;
import com.aresstack.windirectml.runtime.kernels.DirectMlGeluKernel;
import com.aresstack.windirectml.runtime.kernels.DirectMlHeadLayoutKernel;
import com.aresstack.windirectml.runtime.kernels.DirectMlLayerNormKernel;
import com.aresstack.windirectml.runtime.kernels.DirectMlLinearKernel;
import com.aresstack.windirectml.runtime.kernels.GeluKernel;
import com.aresstack.windirectml.windows.D3D12Bindings;
import com.aresstack.windirectml.windows.DxgiBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * One full BERT-style transformer encoder block on DirectML – the
 * model-agnostic generalisation of the original
 * {@code DirectMlMiniLmLayerBlock}. Composes the existing single-purpose
 * DirectML kernels into the canonical pre-LN-attention + post-LN-MLP
 * sub-graph (matches MiniLM, E5, JinaBERT, …):
 * <pre>
 *   x  ─►  Q-Linear   ─►  layout SEQ→HEAD ─┐
 *   x  ─►  K-Linear   ─►  layout SEQ→HEAD ─┼─►  Attention  ─►  layout HEAD→SEQ
 *   x  ─►  V-Linear   ─►  layout SEQ→HEAD ─┘
 *                                                                   │
 *                                              Wo-Linear ◄──────────┘
 *                                                  │
 *   residual1 = x + Wo-out  ─►  LayerNorm₁ ─►  xMid
 *
 *   xMid ─► Inter-Linear ─► GELU ─► Out-Linear ─►  Wmlp-out
 *                                                  │
 *   residual2 = xMid + Wmlp-out ─► LayerNorm₂ ─►  xOut
 * </pre>
 * The block is form-bound: a fresh instance must be created for each
 * unique combination of {@code (seq, hidden, heads, headDim, intermediate)}.
 * <p>
 * <b>Feature levels:</b> GELU goes through
 * {@link GeluKernel#create(DirectMlContextImpl, int)} – fused native on
 * FL ≥ 5.1, composite ERF+IDENTITY+MULTIPLY fallback on older feature
 * levels. All other kernels are FL 2.0 baseline.
 */
public final class DirectMlBertEncoderLayerBlock implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlBertEncoderLayerBlock.class);

    private final DirectMlContextImpl ctx;
    private final int batch;
    private final int seq;
    private final int rows;          // batch * seq – row count for row-wise kernels
    private final int hidden;
    private final int heads;
    private final int headDim;
    private final int intermediate;
    private final float eps;
    private final boolean hasMask;

    private final DirectMlLinearKernel qLinear, kLinear, vLinear, attnOutLinear;
    private final DirectMlLinearKernel mlpInterLinear, mlpOutLinear;
    // Three independent SEQ→HEAD layout kernels for Q/K/V – needed so the
    // submission-coalescing batch (DirectMlGpuBatch) can fire three
    // back-to-back layout dispatches without one overwriting the previous
    // call's descriptor-heap slots before the GPU has consumed them.
    private final DirectMlHeadLayoutKernel layoutSeqToHeadQ;
    private final DirectMlHeadLayoutKernel layoutSeqToHeadK;
    private final DirectMlHeadLayoutKernel layoutSeqToHeadV;
    private final DirectMlHeadLayoutKernel layoutHeadToSeq;
    private final DirectMlAttentionKernel attention;
    private final DirectMlLayerNormKernel attnLayerNorm;
    private final DirectMlLayerNormKernel outLayerNorm;
    private final GeluKernel gelu;
    // Two residual-add kernels (post-attention, post-MLP) – same rationale
    // as the three layout kernels above.
    private final DirectMlAddKernel residualAdd1;
    private final DirectMlAddKernel residualAdd2;

    private final GpuBuffer qBuf, kBuf, vBuf;
    private final GpuBuffer qH, kH, vH, attnH;
    private final GpuBuffer attnMerged;
    private final GpuBuffer attnOutBuf;
    private final GpuBuffer residual1;
    private final GpuBuffer xMid;
    private final GpuBuffer mlpInter;
    private final GpuBuffer mlpOutBuf;
    private final GpuBuffer residual2;

    private boolean closed = false;

    public DirectMlBertEncoderLayerBlock(DirectMlContextImpl ctx,
                                         int seq, int hidden, int heads, int headDim,
                                         int intermediate, float eps, boolean hasMask)
            throws DirectMlRuntimeException {
        this(ctx, 1, seq, hidden, heads, headDim, intermediate, eps, hasMask);
    }

    /**
     * Batched variant – row-wise sub-kernels (Linear/LayerNorm/GELU/Add)
     * see {@code batch * seq} rows; the head-layout and attention kernels
     * receive the real {@code batch} dimension and the mask is
     * {@code [batch, seq]}. The construction with {@code batch = 1} is
     * byte-identical to the legacy 8-arg constructor.
     */
    public DirectMlBertEncoderLayerBlock(DirectMlContextImpl ctx,
                                         int batch, int seq, int hidden, int heads, int headDim,
                                         int intermediate, float eps, boolean hasMask)
            throws DirectMlRuntimeException {
        if (ctx == null || !ctx.isReady()) {
            throw new DirectMlRuntimeException("Context not ready");
        }
        if (batch <= 0 || seq <= 0 || hidden <= 0 || heads <= 0 || headDim <= 0 || intermediate <= 0) {
            throw new IllegalArgumentException(
                    "batch, seq, hidden, heads, headDim, intermediate must be > 0");
        }
        if (hidden != heads * headDim) {
            throw new IllegalArgumentException(
                    "hidden (" + hidden + ") must equal heads*headDim (" + heads + "*" + headDim + ")");
        }
        this.ctx = ctx;
        this.batch = batch;
        this.seq = seq;
        this.rows = batch * seq;
        this.hidden = hidden;
        this.heads = heads;
        this.headDim = headDim;
        this.intermediate = intermediate;
        this.eps = eps;
        this.hasMask = hasMask;

        DirectMlLinearKernel qL = null, kL = null, vL = null, oL = null, mlpI = null, mlpO = null;
        DirectMlHeadLayoutKernel lFwdQ = null, lFwdK = null, lFwdV = null, lBwd = null;
        DirectMlAttentionKernel att = null;
        DirectMlLayerNormKernel ln1 = null, ln2 = null;
        GeluKernel g = null;
        DirectMlAddKernel add1 = null, add2 = null;

        GpuBuffer bq = null, bk = null, bv = null;
        GpuBuffer bqH = null, bkH = null, bvH = null, baH = null;
        GpuBuffer bMerge = null, bOut = null, bRes1 = null, bMid = null;
        GpuBuffer bInter = null, bMlpOut = null, bRes2 = null;

        try {
            float scale = (float) (1.0 / Math.sqrt(headDim));

            qL = new DirectMlLinearKernel(ctx, rows, hidden, hidden, true);
            kL = new DirectMlLinearKernel(ctx, rows, hidden, hidden, true);
            vL = new DirectMlLinearKernel(ctx, rows, hidden, hidden, true);
            oL = new DirectMlLinearKernel(ctx, rows, hidden, hidden, true);
            mlpI = new DirectMlLinearKernel(ctx, rows, hidden, intermediate, true);
            mlpO = new DirectMlLinearKernel(ctx, rows, intermediate, hidden, true);

            lFwdQ = DirectMlHeadLayoutKernel.seqMajorToHeadMajor(ctx, batch, seq, heads, headDim);
            lFwdK = DirectMlHeadLayoutKernel.seqMajorToHeadMajor(ctx, batch, seq, heads, headDim);
            lFwdV = DirectMlHeadLayoutKernel.seqMajorToHeadMajor(ctx, batch, seq, heads, headDim);
            lBwd = DirectMlHeadLayoutKernel.headMajorToSeqMajor(ctx, batch, seq, heads, headDim);

            att = new DirectMlAttentionKernel(ctx, batch, heads, seq, headDim, scale, hasMask);
            ln1 = new DirectMlLayerNormKernel(ctx, rows, hidden, eps);
            ln2 = new DirectMlLayerNormKernel(ctx, rows, hidden, eps);
            g = GeluKernel.create(ctx, rows * intermediate);
            add1 = new DirectMlAddKernel(ctx, rows * hidden);
            add2 = new DirectMlAddKernel(ctx, rows * hidden);

            long hiddenBytes = (long) rows * hidden * Float.BYTES;
            long intermediateBytes = (long) rows * intermediate * Float.BYTES;

            bq = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bk = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bv = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bqH = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bkH = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bvH = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            baH = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bMerge = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bOut = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bRes1 = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bMid = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bInter = ctx.allocateBuffer(intermediateBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bMlpOut = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bRes2 = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);

            this.qLinear = qL;
            this.kLinear = kL;
            this.vLinear = vL;
            this.attnOutLinear = oL;
            this.mlpInterLinear = mlpI;
            this.mlpOutLinear = mlpO;
            this.layoutSeqToHeadQ = lFwdQ;
            this.layoutSeqToHeadK = lFwdK;
            this.layoutSeqToHeadV = lFwdV;
            this.layoutHeadToSeq = lBwd;
            this.attention = att;
            this.attnLayerNorm = ln1;
            this.outLayerNorm = ln2;
            this.gelu = g;
            this.residualAdd1 = add1;
            this.residualAdd2 = add2;

            this.qBuf = bq;
            this.kBuf = bk;
            this.vBuf = bv;
            this.qH = bqH;
            this.kH = bkH;
            this.vH = bvH;
            this.attnH = baH;
            this.attnMerged = bMerge;
            this.attnOutBuf = bOut;
            this.residual1 = bRes1;
            this.xMid = bMid;
            this.mlpInter = bInter;
            this.mlpOutBuf = bMlpOut;
            this.residual2 = bRes2;

            log.info("DirectMlBertEncoderLayerBlock ready: batch={}, seq={}, hidden={} (H={} × D={}), inter={}, hasMask={}",
                    batch, seq, hidden, heads, headDim, intermediate, hasMask);
        } catch (DirectMlRuntimeException | RuntimeException e) {
            closeQuiet(bRes2);
            closeQuiet(bMlpOut);
            closeQuiet(bInter);
            closeQuiet(bMid);
            closeQuiet(bRes1);
            closeQuiet(bOut);
            closeQuiet(bMerge);
            closeQuiet(baH);
            closeQuiet(bvH);
            closeQuiet(bkH);
            closeQuiet(bqH);
            closeQuiet(bv);
            closeQuiet(bk);
            closeQuiet(bq);
            closeQuiet(add2);
            closeQuiet(add1);
            closeQuiet(g);
            closeQuiet(ln2);
            closeQuiet(ln1);
            closeQuiet(att);
            closeQuiet(lBwd);
            closeQuiet(lFwdV);
            closeQuiet(lFwdK);
            closeQuiet(lFwdQ);
            closeQuiet(mlpO);
            closeQuiet(mlpI);
            closeQuiet(oL);
            closeQuiet(vL);
            closeQuiet(kL);
            closeQuiet(qL);
            throw (e instanceof DirectMlRuntimeException d) ? d
                    : new DirectMlRuntimeException("Failed to build DirectMlBertEncoderLayerBlock", e);
        }
    }

    /**
     * Forwards one encoder block on the GPU. {@code xIn} must hold
     * {@code batch * seq * hidden} float32 values laid out row-major as
     * {@code [batch, seq, hidden]} (or, equivalently for the row-wise
     * sub-kernels, {@code [batch * seq, hidden]}). {@code xOut} mirrors
     * that. {@code mask} is {@code [batch, seq]} additive float32 –
     * {@code -1e9} at padding, {@code 0} on valid positions – iff the
     * block was constructed with {@code hasMask=true}.
     */
    public void dispatch(DirectMlTensor xIn,
                         BertGpuLayerWeights w,
                         DirectMlTensor mask,
                         DirectMlTensor xOut) throws DirectMlRuntimeException {
        ensureOpen();
        validateXY(xIn, "xIn");
        validateXY(xOut, "xOut");
        if (hasMask) {
            if (mask == null) throw new DirectMlRuntimeException(
                    "block built with hasMask=true but mask=null");
            long expectedMask = (long) batch * seq;
            if (mask.shape().elementCount() != expectedMask) throw new DirectMlRuntimeException(
                    "mask must hold " + expectedMask + " elements ([batch=" + batch
                            + ", seq=" + seq + "]), got " + mask.shape().elementCount());
        } else if (mask != null) {
            throw new DirectMlRuntimeException("block built with hasMask=false but mask!=null");
        }

        DirectMlTensor qT = hidden2D(qBuf);
        DirectMlTensor kT = hidden2D(kBuf);
        DirectMlTensor vT = hidden2D(vBuf);
        DirectMlTensor qHT = headMajor(qH);
        DirectMlTensor kHT = headMajor(kH);
        DirectMlTensor vHT = headMajor(vH);
        DirectMlTensor attnHT = headMajor(attnH);
        DirectMlTensor attnMergedT = hidden2D(attnMerged);
        DirectMlTensor attnOutT = hidden2D(attnOutBuf);
        DirectMlTensor residual1T = hidden2D(residual1);
        DirectMlTensor xMidT = hidden2D(xMid);
        DirectMlTensor interT = intermediate2D(mlpInter);
        DirectMlTensor mlpOutT = hidden2D(mlpOutBuf);
        DirectMlTensor residual2T = hidden2D(residual2);

        // ── Per-layer command-list coalescing ─────────────────────────────
        // Fold all ~15 sub-ops into ONE command list. Without this we'd
        // pay ~15 ExecuteCommandLists (and ~15 SetDescriptorHeaps flushes)
        // per layer; with it we pay exactly one. The active
        // DirectMlGpuBatch additionally defers the fence wait to the end
        // of the encoder stack.
        MemorySegment dev = ctx.bindings().getD3d12Device();
        MemorySegment q = ctx.bindings().getCommandQueue();
        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment alloc = D3D12Bindings.createCommandAllocator(dev,
                    D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, scratch);
            MemorySegment cl = null;
            try {
                cl = D3D12Bindings.createCommandList(dev,
                        D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, scratch);

                // Q, K, V projections – read xIn, write into independent
                // scratch buffers, can run concurrently on the GPU.
                qLinear.recordOnto(cl, scratch, xIn,
                        weight2D(w.qWeight(), hidden, hidden), bias1D(w.qBias(), hidden), qT);
                kLinear.recordOnto(cl, scratch, xIn,
                        weight2D(w.kWeight(), hidden, hidden), bias1D(w.kBias(), hidden), kT);
                vLinear.recordOnto(cl, scratch, xIn,
                        weight2D(w.vWeight(), hidden, hidden), bias1D(w.vBias(), hidden), vT);
                D3D12Bindings.uavBarrier(cl, scratch);

                // Layout SEQ→HEAD for each of Q, K, V.
                layoutSeqToHeadQ.recordOnto(cl, scratch, qT, qHT);
                layoutSeqToHeadK.recordOnto(cl, scratch, kT, kHT);
                layoutSeqToHeadV.recordOnto(cl, scratch, vT, vHT);
                D3D12Bindings.uavBarrier(cl, scratch);

                // Multi-head attention sub-graph (4 internal stages with
                // their own UAV barriers, see DirectMlAttentionKernel).
                attention.recordOnto(cl, scratch, qHT, kHT, vHT, mask, attnHT,
                        (float) (1.0 / Math.sqrt(headDim)));
                D3D12Bindings.uavBarrier(cl, scratch);

                // Layout HEAD→SEQ.
                layoutHeadToSeq.recordOnto(cl, scratch, attnHT, attnMergedT);
                D3D12Bindings.uavBarrier(cl, scratch);

                // Output projection of the attention block.
                attnOutLinear.recordOnto(cl, scratch, attnMergedT,
                        weight2D(w.attnOutWeight(), hidden, hidden),
                        bias1D(w.attnOutBias(), hidden), attnOutT);
                D3D12Bindings.uavBarrier(cl, scratch);

                // First residual + LayerNorm (post-attention).
                residualAdd1.recordOnto(cl, scratch, xIn, attnOutT, residual1T);
                D3D12Bindings.uavBarrier(cl, scratch);
                attnLayerNorm.recordOnto(cl, scratch, residual1T,
                        bias1D(w.attnLnGamma(), hidden), bias1D(w.attnLnBeta(), hidden),
                        xMidT, eps);
                D3D12Bindings.uavBarrier(cl, scratch);

                // MLP: intermediate Linear → GELU → out Linear.
                mlpInterLinear.recordOnto(cl, scratch, xMidT,
                        weight2D(w.mlpInterWeight(), intermediate, hidden),
                        bias1D(w.mlpInterBias(), intermediate), interT);
                D3D12Bindings.uavBarrier(cl, scratch);
                gelu.recordOnto(cl, scratch, interT, interT);
                D3D12Bindings.uavBarrier(cl, scratch);
                mlpOutLinear.recordOnto(cl, scratch, interT,
                        weight2D(w.mlpOutWeight(), hidden, intermediate),
                        bias1D(w.mlpOutBias(), hidden), mlpOutT);
                D3D12Bindings.uavBarrier(cl, scratch);

                // Second residual + LayerNorm (post-MLP) – writes xOut.
                residualAdd2.recordOnto(cl, scratch, xMidT, mlpOutT, residual2T);
                D3D12Bindings.uavBarrier(cl, scratch);
                outLayerNorm.recordOnto(cl, scratch, residual2T,
                        bias1D(w.outLnGamma(), hidden), bias1D(w.outLnBeta(), hidden),
                        xOut, eps);

                D3D12Bindings.executeOrDefer(dev, q, cl, alloc, scratch);
                DirectMlGpuBatch.recordCoalescedLayerSubmission();
            } finally {
                if (cl != null) DxgiBindings.release(cl);
                DxgiBindings.release(alloc);
            }
        } catch (WindowsNativeException e) {
            throw new DirectMlRuntimeException(
                    "DirectMlBertEncoderLayerBlock.dispatch failed", e);
        }
    }

    private DirectMlTensor hidden2D(GpuBuffer buf) {
        TensorShape s = TensorShape.of(rows, hidden);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private DirectMlTensor intermediate2D(GpuBuffer buf) {
        TensorShape s = TensorShape.of(rows, intermediate);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private DirectMlTensor headMajor(GpuBuffer buf) {
        TensorShape s = TensorShape.of(batch, heads, seq, headDim);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private static DirectMlTensor weight2D(GpuBuffer buf, int outFeatures, int inFeatures) {
        TensorShape s = TensorShape.of(outFeatures, inFeatures);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private static DirectMlTensor bias1D(GpuBuffer buf, int n) {
        TensorShape s = TensorShape.of(n);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private void validateXY(DirectMlTensor t, String name) throws DirectMlRuntimeException {
        if (t == null) throw new DirectMlRuntimeException(name + " tensor is null");
        if (t.dataType() != TensorDataType.FLOAT32) {
            throw new DirectMlRuntimeException(name + " must be FLOAT32");
        }
        long expected = (long) rows * hidden;
        if (t.shape().elementCount() != expected) {
            throw new DirectMlRuntimeException(name + " expected " + expected
                    + " elements ([batch=" + batch + ", seq=" + seq + ", hidden=" + hidden + "]), got "
                    + t.shape().elementCount());
        }
    }

    private void ensureOpen() throws DirectMlRuntimeException {
        if (closed) throw new DirectMlRuntimeException("DirectMlBertEncoderLayerBlock already closed");
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        closeQuiet(residual2);
        closeQuiet(mlpOutBuf);
        closeQuiet(mlpInter);
        closeQuiet(xMid);
        closeQuiet(residual1);
        closeQuiet(attnOutBuf);
        closeQuiet(attnMerged);
        closeQuiet(attnH);
        closeQuiet(vH);
        closeQuiet(kH);
        closeQuiet(qH);
        closeQuiet(vBuf);
        closeQuiet(kBuf);
        closeQuiet(qBuf);
        closeQuiet(residualAdd2);
        closeQuiet(residualAdd1);
        closeQuiet(gelu);
        closeQuiet(outLayerNorm);
        closeQuiet(attnLayerNorm);
        closeQuiet(attention);
        closeQuiet(layoutHeadToSeq);
        closeQuiet(layoutSeqToHeadV);
        closeQuiet(layoutSeqToHeadK);
        closeQuiet(layoutSeqToHeadQ);
        closeQuiet(mlpOutLinear);
        closeQuiet(mlpInterLinear);
        closeQuiet(attnOutLinear);
        closeQuiet(vLinear);
        closeQuiet(kLinear);
        closeQuiet(qLinear);
    }

    private static void closeQuiet(AutoCloseable c) {
        if (c == null) return;
        try {
            c.close();
        } catch (Exception ignored) {
        }
    }

    public int seq() {
        return seq;
    }

    public int batch() {
        return batch;
    }

    public int hidden() {
        return hidden;
    }

    public int heads() {
        return heads;
    }

    public int headDim() {
        return headDim;
    }

    public int intermediate() {
        return intermediate;
    }

    public float eps() {
        return eps;
    }

    public boolean hasMask() {
        return hasMask;
    }
}

