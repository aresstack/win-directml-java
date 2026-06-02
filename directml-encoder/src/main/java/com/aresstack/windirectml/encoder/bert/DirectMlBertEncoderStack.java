package com.aresstack.windirectml.encoder.bert;

import com.aresstack.windirectml.runtime.DirectMlContextImpl;
import com.aresstack.windirectml.runtime.DirectMlGpuBatch;
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
    private final int batch;
    private final int seq;
    private final int rows;          // batch * seq
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
        this(ctx, 1, seq, hidden, heads, headDim, intermediate, numLayers, eps, hasMask);
    }

    /**
     * Batched variant – every dispatched call processes {@code batch}
     * independent sequences of {@code seq} tokens through the same set
     * of weights. The legacy 9-arg constructor delegates here with
     * {@code batch = 1} so existing call sites stay byte-identical.
     */
    public DirectMlBertEncoderStack(DirectMlContextImpl ctx,
                                    int batch, int seq, int hidden, int heads, int headDim,
                                    int intermediate, int numLayers,
                                    float eps, boolean hasMask)
            throws DirectMlRuntimeException {
        if (ctx == null || !ctx.isReady()) {
            throw new DirectMlRuntimeException("Context not ready");
        }
        if (batch <= 0 || seq <= 0 || hidden <= 0 || heads <= 0 || headDim <= 0
                || intermediate <= 0 || numLayers <= 0) {
            throw new IllegalArgumentException(
                    "batch, seq, hidden, heads, headDim, intermediate, numLayers must be > 0");
        }
        if (hidden != heads * headDim) {
            throw new IllegalArgumentException(
                    "hidden (" + hidden + ") must equal heads*headDim ("
                            + heads + "*" + headDim + ")");
        }
        this.ctx = ctx;
        this.batch = batch;
        this.seq = seq;
        this.rows = batch * seq;
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
            embLn = new DirectMlLayerNormKernel(ctx, rows, hidden, eps);

            long hiddenBytes = (long) rows * hidden * Float.BYTES;
            sA = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            if (numLayers >= 2) {
                sB = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            }

            for (int i = 0; i < numLayers; i++) {
                blockList.add(new DirectMlBertEncoderLayerBlock(
                        ctx, batch, seq, hidden, heads, headDim, intermediate, eps, hasMask));
            }

            this.embeddingLayerNorm = embLn;
            this.blocks = List.copyOf(blockList);
            this.scratchA = sA;
            this.scratchB = sB;

            log.info("DirectMlBertEncoderStack ready: batch={}, seq={}, hidden={} (H={} × D={}), inter={}, layers={}, hasMask={}",
                    batch, seq, hidden, heads, headDim, intermediate, numLayers, hasMask);
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
     * @param xIn          pre-summed embeddings {@code [batch * seq, hidden]} F32.
     * @param embLnGamma   embedding-LN gain {@code [hidden]}.
     * @param embLnBeta    embedding-LN bias {@code [hidden]}.
     * @param layerWeights per-layer weights ({@code size() == numLayers}).
     * @param mask         additive float mask {@code [batch * seq]} (0 / -1e9) if
     *                     {@code hasMask=true}, else null.
     * @param xOut         destination {@code [batch * seq, hidden]} after the
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
            long expectedMask = (long) batch * seq;
            if (mask.shape().elementCount() != expectedMask) throw new DirectMlRuntimeException(
                    "mask must hold " + expectedMask + " elements ([batch=" + batch
                            + ", seq=" + seq + "]), got " + mask.shape().elementCount());
        } else if (mask != null) {
            throw new DirectMlRuntimeException("stack built with hasMask=false but mask!=null");
        }

        // ── Submission coalescing ──────────────────────────────────────────
        // Every per-kernel dispatch below would normally pay its own fence
        // wait inside D3D12Bindings.executeAndWait (~80–100 fences/text for
        // MiniLM, which dominates the ~44 ms/text baseline). By wrapping the
        // whole forward pass in a DirectMlGpuBatch, all kernels submit their
        // command lists fire-and-forget; a single fence drain happens once
        // in the try-with-resources close at the end.
        try (DirectMlGpuBatch batch = DirectMlGpuBatch.begin(ctx.bindings())) {
            // 1. Embedding-LN: xIn → scratchA
            DirectMlTensor curT = hidden2D(scratchA);
            embeddingLayerNorm.dispatch(xIn, embLnGamma, embLnBeta, curT, eps);

            // 2. Layer cascade with ping-pong
            GpuBuffer curBuf = scratchA;
            for (int i = 0; i < numLayers; i++) {
                boolean last = (i == numLayers - 1);
                GpuBuffer nextBuf = last ? null : (curBuf == scratchA ? scratchB : scratchA);
                DirectMlTensor inT = hidden2D(curBuf);
                DirectMlTensor outT = last ? xOut : hidden2D(nextBuf);
                blocks.get(i).dispatch(inT, layerWeights.get(i), mask, outT);
                curBuf = nextBuf;
            }
            if (log.isDebugEnabled()) {
                log.debug("DirectMlBertEncoderStack.dispatch coalesced {} GPU submissions",
                        batch.submissions());
            }
        }
    }

    private DirectMlTensor hidden2D(GpuBuffer buf) {
        TensorShape s = TensorShape.of(rows, hidden);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private void validateHidden(DirectMlTensor t, String name) throws DirectMlRuntimeException {
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

    public int numLayers() {
        return numLayers;
    }

    public float eps() {
        return eps;
    }

    public boolean hasMask() {
        return hasMask;
    }
}

