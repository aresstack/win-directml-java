package com.aresstack.windirectml.encoder.minilm;

import com.aresstack.windirectml.runtime.DirectMlContextImpl;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;
import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.GpuBuffer;
import com.aresstack.windirectml.runtime.TensorDataType;
import com.aresstack.windirectml.runtime.TensorLayout;
import com.aresstack.windirectml.runtime.TensorShape;
import com.aresstack.windirectml.runtime.kernels.DirectMlLayerNormKernel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Volle MiniLM/BERT-Encoder-Pipeline auf DirectML – Embedding-LayerNorm
 * gefolgt von {@code N} aufeinanderfolgenden {@link DirectMlMiniLmLayerBlock}
 * Stufen.
 * <p>
 * Erwartet als Eingang die schon summierten Embeddings
 * ({@code word + position + tokenType}) im Shape {@code [seq, hidden]}.
 * Der Lookup ist eine reine Speicheroperation und bleibt deshalb bewusst
 * auf der CPU – das spart einen GPU-Gather und hält die GPU-API
 * vorhersehbar typisiert (FL 2.0 hat kein DML_OPERATOR_GATHER vor 5.0).
 * <p>
 * Buffer-Choreographie für {@code numLayers} Stufen:
 * <pre>
 *   xIn ─► embeddingLN ─► scratchA ─► block[0] ─► scratchB ─► block[1] ─► scratchA …
 *                                                                       └─► xOut
 * </pre>
 * Wenn {@code numLayers == 1}, schreibt der einzige Block direkt nach
 * {@code xOut}. {@code scratchB} wird nur allokiert, wenn {@code numLayers >= 2}.
 */
public final class DirectMlMiniLmEncoderStack implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlMiniLmEncoderStack.class);

    private final DirectMlContextImpl ctx;
    private final int seq;
    private final int hidden;
    private final int heads;
    private final int headDim;
    private final int intermediate;
    private final int numLayers;
    private final float eps;
    private final boolean hasMask;

    private final DirectMlLayerNormKernel embeddingLayerNorm;
    private final List<DirectMlMiniLmLayerBlock> blocks;

    private final GpuBuffer scratchA;       // [seq, hidden]
    private final GpuBuffer scratchB;       // [seq, hidden] – nur bei numLayers>=2

    private boolean closed = false;

    public DirectMlMiniLmEncoderStack(DirectMlContextImpl ctx,
                                      int seq, int hidden, int heads, int headDim,
                                      int intermediate, int numLayers,
                                      float eps, boolean hasMask)
            throws DirectMlRuntimeException {
        if (ctx == null || !ctx.isReady()) {
            throw new DirectMlRuntimeException("Context not ready");
        }
        if (seq <= 0 || hidden <= 0 || heads <= 0 || headDim <= 0
                || intermediate <= 0 || numLayers <= 0) {
            throw new IllegalArgumentException(
                    "seq, hidden, heads, headDim, intermediate, numLayers must be > 0");
        }
        if (hidden != heads * headDim) {
            throw new IllegalArgumentException(
                    "hidden (" + hidden + ") must equal heads*headDim ("
                            + heads + "*" + headDim + ")");
        }
        this.ctx = ctx;
        this.seq = seq;
        this.hidden = hidden;
        this.heads = heads;
        this.headDim = headDim;
        this.intermediate = intermediate;
        this.numLayers = numLayers;
        this.eps = eps;
        this.hasMask = hasMask;

        DirectMlLayerNormKernel embLn = null;
        List<DirectMlMiniLmLayerBlock> blockList = new ArrayList<>(numLayers);
        GpuBuffer sA = null, sB = null;

        try {
            embLn = new DirectMlLayerNormKernel(ctx, seq, hidden, eps);

            long hiddenBytes = (long) seq * hidden * Float.BYTES;
            sA = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            if (numLayers >= 2) {
                sB = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            }

            for (int i = 0; i < numLayers; i++) {
                blockList.add(new DirectMlMiniLmLayerBlock(
                        ctx, seq, hidden, heads, headDim, intermediate, eps, hasMask));
            }

            this.embeddingLayerNorm = embLn;
            this.blocks = List.copyOf(blockList);
            this.scratchA = sA;
            this.scratchB = sB;

            log.info("DirectMlMiniLmEncoderStack ready: seq={}, hidden={} (H={} × D={}), inter={}, layers={}, hasMask={}",
                    seq, hidden, heads, headDim, intermediate, numLayers, hasMask);
        } catch (DirectMlRuntimeException | RuntimeException e) {
            for (int i = blockList.size() - 1; i >= 0; i--) closeQuiet(blockList.get(i));
            closeQuiet(sB);
            closeQuiet(sA);
            closeQuiet(embLn);
            throw (e instanceof DirectMlRuntimeException d) ? d
                    : new DirectMlRuntimeException("Failed to build DirectMlMiniLmEncoderStack", e);
        }
    }

    /**
     * Dispatch der gesamten Encoder-Pipeline.
     * @param xIn        Pre-Embedding-Sum {@code [seq, hidden]} (word+pos+tokenType, F32).
     * @param embLnGamma Embedding-LN-Gain {@code [hidden]}.
     * @param embLnBeta  Embedding-LN-Bias {@code [hidden]}.
     * @param layerWeights Per-Layer-Gewichte; {@code size()} muss {@link #numLayers()} entsprechen.
     * @param mask       Additive Float-Maske {@code [seq]} (0/-1e9) wenn {@code hasMask==true}, sonst {@code null}.
     * @param xOut       Ziel-Tensor {@code [seq, hidden]} für den letzten LayerNorm-Output.
     */
    public void dispatch(DirectMlTensor xIn,
                         DirectMlTensor embLnGamma,
                         DirectMlTensor embLnBeta,
                         List<DirectMlMiniLmLayerBlock.LayerWeights> layerWeights,
                         DirectMlTensor mask,
                         DirectMlTensor xOut) throws DirectMlRuntimeException {
        ensureOpen();
        Objects.requireNonNull(layerWeights, "layerWeights");
        if (layerWeights.size() != numLayers) {
            throw new DirectMlRuntimeException("layerWeights.size()=" + layerWeights.size()
                    + " but stack expects " + numLayers + " layers");
        }
        validateHidden(xIn, "xIn");
        validateHidden(xOut, "xOut");
        if (hasMask) {
            if (mask == null) throw new DirectMlRuntimeException(
                    "stack built with hasMask=true but mask=null");
            if (mask.shape().elementCount() != seq) throw new DirectMlRuntimeException(
                    "mask must hold " + seq + " elements, got " + mask.shape().elementCount());
        } else if (mask != null) {
            throw new DirectMlRuntimeException("stack built with hasMask=false but mask!=null");
        }

        // ── 1. Embedding-LN: xIn → scratchA ────────────────────────────
        DirectMlTensor curT = hidden2D(scratchA);
        embeddingLayerNorm.dispatch(xIn, embLnGamma, embLnBeta, curT, eps);

        // ── 2. Layer-Kaskade mit Ping-Pong ─────────────────────────────
        GpuBuffer curBuf = scratchA;
        for (int i = 0; i < numLayers; i++) {
            boolean last = (i == numLayers - 1);
            GpuBuffer nextBuf = last ? null : (curBuf == scratchA ? scratchB : scratchA);
            DirectMlTensor inT  = hidden2D(curBuf);
            DirectMlTensor outT = last ? xOut : hidden2D(nextBuf);
            blocks.get(i).dispatch(inT, layerWeights.get(i), mask, outT);
            curBuf = nextBuf;
        }
    }

    // ── helpers ─────────────────────────────────────────────────────────

    private DirectMlTensor hidden2D(GpuBuffer buf) {
        TensorShape s = TensorShape.of(seq, hidden);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private void validateHidden(DirectMlTensor t, String name) throws DirectMlRuntimeException {
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
        if (closed) throw new DirectMlRuntimeException("DirectMlMiniLmEncoderStack already closed");
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        for (int i = blocks.size() - 1; i >= 0; i--) closeQuiet(blocks.get(i));
        closeQuiet(scratchB);
        closeQuiet(scratchA);
        closeQuiet(embeddingLayerNorm);
    }

    private static void closeQuiet(AutoCloseable c) {
        if (c == null) return;
        try { c.close(); } catch (Exception ignored) { /* best-effort */ }
    }

    public int seq()          { return seq; }
    public int hidden()       { return hidden; }
    public int heads()        { return heads; }
    public int headDim()      { return headDim; }
    public int intermediate() { return intermediate; }
    public int numLayers()    { return numLayers; }
    public float eps()        { return eps; }
    public boolean hasMask()  { return hasMask; }
}

