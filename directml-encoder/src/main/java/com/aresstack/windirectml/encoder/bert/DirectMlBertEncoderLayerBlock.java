package com.aresstack.windirectml.encoder.bert;

import com.aresstack.windirectml.runtime.DirectMlContextImpl;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
    private final int seq;
    private final int hidden;
    private final int heads;
    private final int headDim;
    private final int intermediate;
    private final float eps;
    private final boolean hasMask;

    private final DirectMlLinearKernel qLinear, kLinear, vLinear, attnOutLinear;
    private final DirectMlLinearKernel mlpInterLinear, mlpOutLinear;
    private final DirectMlHeadLayoutKernel layoutSeqToHead;
    private final DirectMlHeadLayoutKernel layoutHeadToSeq;
    private final DirectMlAttentionKernel attention;
    private final DirectMlLayerNormKernel attnLayerNorm;
    private final DirectMlLayerNormKernel outLayerNorm;
    private final GeluKernel gelu;
    private final DirectMlAddKernel residualAdd;

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
        if (ctx == null || !ctx.isReady()) {
            throw new DirectMlRuntimeException("Context not ready");
        }
        if (seq <= 0 || hidden <= 0 || heads <= 0 || headDim <= 0 || intermediate <= 0) {
            throw new IllegalArgumentException("seq, hidden, heads, headDim, intermediate must be > 0");
        }
        if (hidden != heads * headDim) {
            throw new IllegalArgumentException(
                    "hidden (" + hidden + ") must equal heads*headDim (" + heads + "*" + headDim + ")");
        }
        this.ctx = ctx;
        this.seq = seq;
        this.hidden = hidden;
        this.heads = heads;
        this.headDim = headDim;
        this.intermediate = intermediate;
        this.eps = eps;
        this.hasMask = hasMask;

        DirectMlLinearKernel qL = null, kL = null, vL = null, oL = null, mlpI = null, mlpO = null;
        DirectMlHeadLayoutKernel lFwd = null, lBwd = null;
        DirectMlAttentionKernel att = null;
        DirectMlLayerNormKernel ln1 = null, ln2 = null;
        GeluKernel g = null;
        DirectMlAddKernel add = null;

        GpuBuffer bq = null, bk = null, bv = null;
        GpuBuffer bqH = null, bkH = null, bvH = null, baH = null;
        GpuBuffer bMerge = null, bOut = null, bRes1 = null, bMid = null;
        GpuBuffer bInter = null, bMlpOut = null, bRes2 = null;

        try {
            float scale = (float) (1.0 / Math.sqrt(headDim));

            qL    = new DirectMlLinearKernel(ctx, seq, hidden, hidden, true);
            kL    = new DirectMlLinearKernel(ctx, seq, hidden, hidden, true);
            vL    = new DirectMlLinearKernel(ctx, seq, hidden, hidden, true);
            oL    = new DirectMlLinearKernel(ctx, seq, hidden, hidden, true);
            mlpI  = new DirectMlLinearKernel(ctx, seq, hidden, intermediate, true);
            mlpO  = new DirectMlLinearKernel(ctx, seq, intermediate, hidden, true);

            lFwd  = DirectMlHeadLayoutKernel.seqMajorToHeadMajor(ctx, seq, heads, headDim);
            lBwd  = DirectMlHeadLayoutKernel.headMajorToSeqMajor(ctx, seq, heads, headDim);

            att   = new DirectMlAttentionKernel(ctx, 1, heads, seq, headDim, scale, hasMask);
            ln1   = new DirectMlLayerNormKernel(ctx, seq, hidden, eps);
            ln2   = new DirectMlLayerNormKernel(ctx, seq, hidden, eps);
            g     = GeluKernel.create(ctx, seq * intermediate);
            add   = new DirectMlAddKernel(ctx, seq * hidden);

            long hiddenBytes       = (long) seq * hidden * Float.BYTES;
            long intermediateBytes = (long) seq * intermediate * Float.BYTES;

            bq      = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bk      = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bv      = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bqH     = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bkH     = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bvH     = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            baH     = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bMerge  = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bOut    = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bRes1   = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bMid    = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bInter  = ctx.allocateBuffer(intermediateBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bMlpOut = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            bRes2   = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);

            this.qLinear = qL; this.kLinear = kL; this.vLinear = vL; this.attnOutLinear = oL;
            this.mlpInterLinear = mlpI; this.mlpOutLinear = mlpO;
            this.layoutSeqToHead = lFwd; this.layoutHeadToSeq = lBwd;
            this.attention = att;
            this.attnLayerNorm = ln1; this.outLayerNorm = ln2;
            this.gelu = g;
            this.residualAdd = add;

            this.qBuf = bq; this.kBuf = bk; this.vBuf = bv;
            this.qH = bqH; this.kH = bkH; this.vH = bvH; this.attnH = baH;
            this.attnMerged = bMerge; this.attnOutBuf = bOut;
            this.residual1 = bRes1; this.xMid = bMid;
            this.mlpInter = bInter; this.mlpOutBuf = bMlpOut; this.residual2 = bRes2;

            log.info("DirectMlBertEncoderLayerBlock ready: seq={}, hidden={} (H={} × D={}), inter={}, hasMask={}",
                    seq, hidden, heads, headDim, intermediate, hasMask);
        } catch (DirectMlRuntimeException | RuntimeException e) {
            closeQuiet(bRes2); closeQuiet(bMlpOut); closeQuiet(bInter);
            closeQuiet(bMid); closeQuiet(bRes1); closeQuiet(bOut); closeQuiet(bMerge);
            closeQuiet(baH); closeQuiet(bvH); closeQuiet(bkH); closeQuiet(bqH);
            closeQuiet(bv); closeQuiet(bk); closeQuiet(bq);
            closeQuiet(add); closeQuiet(g); closeQuiet(ln2); closeQuiet(ln1); closeQuiet(att);
            closeQuiet(lBwd); closeQuiet(lFwd);
            closeQuiet(mlpO); closeQuiet(mlpI); closeQuiet(oL); closeQuiet(vL); closeQuiet(kL); closeQuiet(qL);
            throw (e instanceof DirectMlRuntimeException d) ? d
                    : new DirectMlRuntimeException("Failed to build DirectMlBertEncoderLayerBlock", e);
        }
    }

    /**
     * Forwards one encoder block on the GPU. {@code xIn} must be a
     * {@code [seq, hidden]} float32 tensor; {@code xOut} receives the
     * same shape after the full block. {@code mask} is {@code [seq]}
     * additive float32 (-1e9 at padding, 0 elsewhere) iff the block
     * was constructed with {@code hasMask=true}; otherwise pass null.
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
            if (mask.shape().elementCount() != seq) throw new DirectMlRuntimeException(
                    "mask must hold " + seq + " elements, got " + mask.shape().elementCount());
        } else if (mask != null) {
            throw new DirectMlRuntimeException("block built with hasMask=false but mask!=null");
        }

        DirectMlTensor qT = hidden2D(qBuf);
        DirectMlTensor kT = hidden2D(kBuf);
        DirectMlTensor vT = hidden2D(vBuf);
        qLinear.dispatch(xIn, weight2D(w.qWeight(), hidden, hidden), bias1D(w.qBias(), hidden), qT);
        kLinear.dispatch(xIn, weight2D(w.kWeight(), hidden, hidden), bias1D(w.kBias(), hidden), kT);
        vLinear.dispatch(xIn, weight2D(w.vWeight(), hidden, hidden), bias1D(w.vBias(), hidden), vT);

        DirectMlTensor qHT = headMajor(qH);
        DirectMlTensor kHT = headMajor(kH);
        DirectMlTensor vHT = headMajor(vH);
        layoutSeqToHead.dispatch(qT, qHT);
        layoutSeqToHead.dispatch(kT, kHT);
        layoutSeqToHead.dispatch(vT, vHT);

        DirectMlTensor attnHT = headMajor(attnH);
        attention.dispatch(qHT, kHT, vHT, mask, attnHT, (float) (1.0 / Math.sqrt(headDim)));

        DirectMlTensor attnMergedT = hidden2D(attnMerged);
        layoutHeadToSeq.dispatch(attnHT, attnMergedT);

        DirectMlTensor attnOutT = hidden2D(attnOutBuf);
        attnOutLinear.dispatch(attnMergedT,
                weight2D(w.attnOutWeight(), hidden, hidden),
                bias1D(w.attnOutBias(), hidden), attnOutT);

        DirectMlTensor residual1T = hidden2D(residual1);
        residualAdd.dispatch(xIn, attnOutT, residual1T);
        DirectMlTensor xMidT = hidden2D(xMid);
        attnLayerNorm.dispatch(residual1T, bias1D(w.attnLnGamma(), hidden),
                bias1D(w.attnLnBeta(), hidden), xMidT, eps);

        DirectMlTensor interT = intermediate2D(mlpInter);
        mlpInterLinear.dispatch(xMidT, weight2D(w.mlpInterWeight(), intermediate, hidden),
                bias1D(w.mlpInterBias(), intermediate), interT);
        gelu.dispatch(interT, interT);
        DirectMlTensor mlpOutT = hidden2D(mlpOutBuf);
        mlpOutLinear.dispatch(interT, weight2D(w.mlpOutWeight(), hidden, intermediate),
                bias1D(w.mlpOutBias(), hidden), mlpOutT);

        DirectMlTensor residual2T = hidden2D(residual2);
        residualAdd.dispatch(xMidT, mlpOutT, residual2T);
        outLayerNorm.dispatch(residual2T, bias1D(w.outLnGamma(), hidden),
                bias1D(w.outLnBeta(), hidden), xOut, eps);
    }

    private DirectMlTensor hidden2D(GpuBuffer buf) {
        TensorShape s = TensorShape.of(seq, hidden);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private DirectMlTensor intermediate2D(GpuBuffer buf) {
        TensorShape s = TensorShape.of(seq, intermediate);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private DirectMlTensor headMajor(GpuBuffer buf) {
        TensorShape s = TensorShape.of(1, heads, seq, headDim);
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
        if (t.shape().elementCount() != (long) seq * hidden) {
            throw new DirectMlRuntimeException(name + " expected " + (seq * hidden)
                    + " elements ([seq=" + seq + ", hidden=" + hidden + "]), got "
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
        closeQuiet(residual2); closeQuiet(mlpOutBuf); closeQuiet(mlpInter);
        closeQuiet(xMid); closeQuiet(residual1); closeQuiet(attnOutBuf); closeQuiet(attnMerged);
        closeQuiet(attnH); closeQuiet(vH); closeQuiet(kH); closeQuiet(qH);
        closeQuiet(vBuf); closeQuiet(kBuf); closeQuiet(qBuf);
        closeQuiet(residualAdd); closeQuiet(gelu);
        closeQuiet(outLayerNorm); closeQuiet(attnLayerNorm);
        closeQuiet(attention);
        closeQuiet(layoutHeadToSeq); closeQuiet(layoutSeqToHead);
        closeQuiet(mlpOutLinear); closeQuiet(mlpInterLinear);
        closeQuiet(attnOutLinear); closeQuiet(vLinear); closeQuiet(kLinear); closeQuiet(qLinear);
    }

    private static void closeQuiet(AutoCloseable c) {
        if (c == null) return;
        try { c.close(); } catch (Exception ignored) {}
    }

    public int seq()          { return seq; }
    public int hidden()       { return hidden; }
    public int heads()        { return heads; }
    public int headDim()      { return headDim; }
    public int intermediate() { return intermediate; }
    public float eps()        { return eps; }
    public boolean hasMask()  { return hasMask; }
}

