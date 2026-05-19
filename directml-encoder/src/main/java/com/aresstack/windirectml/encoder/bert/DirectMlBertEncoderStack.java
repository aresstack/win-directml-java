package com.aresstack.windirectml.encoder.bert;

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
 * Full BERT-style encoder pipeline on DirectML – embedding-LayerNorm
 * followed by {@code numLayers} stacked {@link DirectMlBertEncoderLayerBlock}
 * stages. Model-agnostic generalisation of the original
 * {@code DirectMlMiniLmEncoderStack}.
 * <p>
 * Expects input {@code [seq, hidden]} containing the already-summed
 * embeddings ({@code word + position + tokenType}) – see
 * {@link BertEmbeddingLookup}. The embedding gather stays on the CPU
 * deliberately: it is a pure memory operation and avoids requiring a
 * {@code DML_OPERATOR_GATHER} (FL 5.0+) that is not available on every
 * shipping in-box {@code DirectML.dll}.
 * <p>
 * Buffer choreography (ping-pong with two scratch buffers):
 * <pre>
 *   xIn ─► embeddingLN ─► scratchA ─► block[0] ─► scratchB ─► block[1] ─► scratchA …
 *                                                                       └─► xOut
 * </pre>
 */
public final class DirectMlBertEncoderStack implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlBertEncoderStack.class);

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
    private final List<DirectMlBertEncoderLayerBlock> blocks;

    private final GpuBuffer scratchA;
    private final GpuBuffer scratchB;

    private boolean closed = false;

    public DirectMlBertEncoderStack(DirectMlContextImpl ctx,
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
        List<DirectMlBertEncoderLayerBlock> blockList = new ArrayList<>(numLayers);
        GpuBuffer sA = null, sB = null;

        try {
            embLn = new DirectMlLayerNormKernel(ctx, seq, hidden, eps);

            long hiddenBytes = (long) seq * hidden * Float.BYTES;
            sA = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            if (numLayers >= 2) {
                sB = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            }

            for (int i = 0; i < numLayers; i++) {
                blockList.add(new DirectMlBertEncoderLayerBlock(
                        ctx, seq, hidden, heads, headDim, intermediate, eps, hasMask));
            }

            this.embeddingLayerNorm = embLn;
            this.blocks = List.copyOf(blockList);
            this.scratchA = sA;
            this.scratchB = sB;

            log.info("DirectMlBertEncoderStack ready: seq={}, hidden={} (H={} × D={}), inter={}, layers={}, hasMask={}",
                    seq, hidden, heads, headDim, intermediate, numLayers, hasMask);
        } catch (DirectMlRuntimeException | RuntimeException e) {
            for (int i = blockList.size() - 1; i >= 0; i--) closeQuiet(blockList.get(i));
            closeQuiet(sB);
            closeQuiet(sA);
            closeQuiet(embLn);
            throw (e instanceof DirectMlRuntimeException d) ? d
                    : new DirectMlRuntimeException("Failed to build DirectMlBertEncoderStack", e);
        }
    }

    /**
     * Dispatch the full encoder pipeline.
     *
     * @param xIn          pre-summed embeddings {@code [seq, hidden]} F32.
     * @param embLnGamma   embedding-LN gain {@code [hidden]}.
     * @param embLnBeta    embedding-LN bias {@code [hidden]}.
     * @param layerWeights per-layer weights ({@code size() == numLayers}).
     * @param mask         additive float mask {@code [seq]} (0 / -1e9) if
     *                     {@code hasMask=true}, else null.
     * @param xOut         destination {@code [seq, hidden]} after the
     *                     last LayerNorm.
     */
    public void dispatch(DirectMlTensor xIn,
                         DirectMlTensor embLnGamma,
                         DirectMlTensor embLnBeta,
                         List<BertGpuLayerWeights> layerWeights,
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

        // 1. Embedding-LN: xIn → scratchA
        DirectMlTensor curT = hidden2D(scratchA);
        embeddingLayerNorm.dispatch(xIn, embLnGamma, embLnBeta, curT, eps);

        // 2. Layer cascade with ping-pong
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
        if (closed) throw new DirectMlRuntimeException("DirectMlBertEncoderStack already closed");
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
        try { c.close(); } catch (Exception ignored) {}
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

