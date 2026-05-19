package com.aresstack.windirectml.encoder.bert;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.encoder.EncoderTokenizer;
import com.aresstack.windirectml.runtime.CpuTensor;
import com.aresstack.windirectml.runtime.DirectMlContextImpl;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;
import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.GpuBuffer;
import com.aresstack.windirectml.runtime.TensorDataType;
import com.aresstack.windirectml.runtime.TensorLayout;
import com.aresstack.windirectml.runtime.TensorShape;
import com.aresstack.windirectml.runtime.kernels.DirectMlL2NormalizeKernel;
import com.aresstack.windirectml.runtime.kernels.DirectMlMeanPoolKernel;
import com.aresstack.windirectml.windows.DirectMlBindings;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Generic DirectML BERT-style sentence-embedding encoder. Drives the
 * shared {@link DirectMlBertEncoderStack} + GPU mean-pool + GPU L2
 * normalisation with any {@link BertEncoderConfig} – MiniLM, E5-v2,
 * BGE, … reuse the exact same compute graph.
 * <p>
 * Pipeline (identical to {@code DirectMlMiniLmEncoder}):
 * <pre>
 *   text → tokenizer
 *        → CPU embedding lookup (BertEmbeddingLookup)
 *        → upload to GPU
 *        → DirectMlBertEncoderStack (embLN + N layer)
 *        → DirectMlMeanPoolKernel  (GEMM, mean weights pre-normalised on CPU)
 *        → DirectMlL2NormalizeKernel (FL-1.0 composite)
 *        → download [H]
 *        → EmbeddingVector
 * </pre>
 * Pad-buckets {@code S ∈ {64, 128, 256, 512}} coalesce arbitrary
 * sequence lengths onto a small set of form-bound stacks; the cache
 * therefore holds at most four entries per encoder.
 * <p>
 * The encoder is created via
 * {@link #build(DirectMlContextImpl, boolean, BertEncoderConfig,
 * BertCpuEncoderWeights, EncoderTokenizer)} – it consumes a fully
 * loaded weight bundle plus the matching tokenizer. Family-specific
 * thin loaders (e.g. {@code DirectMlMiniLmEncoder.load(modelDir)})
 * provide the model-aware shimming.
 */
public final class DirectMlBertEncoder implements EmbeddingModel, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlBertEncoder.class);

    /** Default pad-buckets. {@code maxPositionEmbeddings} caps the largest reusable bucket. */
    private static final int[] PAD_BUCKETS = {64, 128, 256, 512};

    /** Defensive-copy accessor for the configured pad-buckets. */
    public static int[] buckets() { return PAD_BUCKETS.clone(); }

    /** Smallest bucket {@code b ≥ seqLen}, else {@code seqLen} itself. */
    public static int bucketFor(int seqLen) {
        for (int b : PAD_BUCKETS) {
            if (b >= seqLen) return b;
        }
        return seqLen;
    }

    private final BertEncoderConfig cfg;
    private final BertCpuEncoderWeights weights;
    private final EncoderTokenizer tokenizer;
    private final DirectMlContextImpl ctx;
    private final boolean ownsCtx;

    private final List<GpuBuffer> ownedGpuBuffers = new ArrayList<>();
    private final GpuBuffer embLnGammaBuf;
    private final GpuBuffer embLnBetaBuf;
    private final List<BertGpuLayerWeights> gpuLayers;
    private final DirectMlL2NormalizeKernel l2Kernel;
    private final GpuBuffer normalizedBuf;

    private final Map<Integer, StackEntry> stackCache = new HashMap<>();
    private volatile boolean ready;

    private record StackEntry(DirectMlBertEncoderStack stack,
                              GpuBuffer xIn, GpuBuffer xOut,
                              GpuBuffer mask, GpuBuffer meanWeights, GpuBuffer pooled,
                              DirectMlMeanPoolKernel meanPool) implements AutoCloseable {
        @Override public void close() {
            try { meanPool.close();    } catch (Exception ignored) {}
            try { stack.close();       } catch (Exception ignored) {}
            try { xIn.close();         } catch (Exception ignored) {}
            try { xOut.close();        } catch (Exception ignored) {}
            try { mask.close();        } catch (Exception ignored) {}
            try { meanWeights.close(); } catch (Exception ignored) {}
            try { pooled.close();      } catch (Exception ignored) {}
        }
    }

    /**
     * Construct an encoder over the given context. {@code ownsCtx=true}
     * means {@link #close()} also closes the context (used by the
     * family-specific {@code load(Path)} convenience loaders).
     */
    public static DirectMlBertEncoder build(DirectMlContextImpl ctx,
                                            boolean ownsCtx,
                                            BertEncoderConfig cfg,
                                            BertCpuEncoderWeights weights,
                                            EncoderTokenizer tokenizer)
            throws EmbeddingException {
        return new DirectMlBertEncoder(ctx, ownsCtx, cfg, weights, tokenizer);
    }

    private DirectMlBertEncoder(DirectMlContextImpl ctx,
                                boolean ownsCtx,
                                BertEncoderConfig cfg,
                                BertCpuEncoderWeights weights,
                                EncoderTokenizer tokenizer) throws EmbeddingException {
        this.ctx = Objects.requireNonNull(ctx);
        this.ownsCtx = ownsCtx;
        this.cfg = Objects.requireNonNull(cfg);
        this.weights = Objects.requireNonNull(weights);
        this.tokenizer = Objects.requireNonNull(tokenizer);
        this.cfg.validate();

        try {
            if (!WindowsBindings.isSupported()) {
                throw new EmbeddingException("DirectML requires Windows + D3D12 on this host");
            }
            if (!ctx.isReady() || !ctx.bindings().hasDirectMl()) {
                throw new EmbeddingException("DirectML context not ready");
            }

            int H = cfg.hiddenSize();
            int I = cfg.intermediateSize();

            embLnGammaBuf = uploadVec(weights.embLnGamma, H);
            embLnBetaBuf  = uploadVec(weights.embLnBeta,  H);

            List<BertGpuLayerWeights> built = new ArrayList<>(weights.layers.size());
            for (BertCpuLayerWeights l : weights.layers) {
                built.add(new BertGpuLayerWeights(
                        uploadMat(l.qWeight(), H, H), uploadVec(l.qBias(), H),
                        uploadMat(l.kWeight(), H, H), uploadVec(l.kBias(), H),
                        uploadMat(l.vWeight(), H, H), uploadVec(l.vBias(), H),
                        uploadMat(l.attnOutWeight(), H, H), uploadVec(l.attnOutBias(), H),
                        uploadVec(l.attnLnGamma(), H), uploadVec(l.attnLnBeta(), H),
                        uploadMat(l.mlpInterWeight(), I, H), uploadVec(l.mlpInterBias(), I),
                        uploadMat(l.mlpOutWeight(),   H, I), uploadVec(l.mlpOutBias(), H),
                        uploadVec(l.outLnGamma(), H), uploadVec(l.outLnBeta(), H)));
            }
            this.gpuLayers = List.copyOf(built);

            DirectMlL2NormalizeKernel builtL2 = null;
            GpuBuffer builtNorm = null;
            try {
                builtL2 = new DirectMlL2NormalizeKernel(ctx, H, 1e-12f);
                builtNorm = ctx.allocateBuffer((long) H * Float.BYTES,
                        GpuBuffer.BufferUsage.ACTIVATION);
            } catch (DirectMlRuntimeException | RuntimeException e) {
                if (builtL2 != null) try { builtL2.close(); } catch (Exception ignored) {}
                if (builtNorm != null) try { builtNorm.close(); } catch (Exception ignored) {}
                throw e;
            }
            this.l2Kernel = builtL2;
            this.normalizedBuf = builtNorm;
            ownedGpuBuffers.add(normalizedBuf);

            this.ready = true;
            log.info("DirectMlBertEncoder ready: model={}, layers={}, hidden={}, heads={}, inter={}, FL={}",
                    cfg.modelName(), cfg.numLayers(), cfg.hiddenSize(),
                    cfg.numHeads(), cfg.intermediateSize(),
                    DirectMlBindings.formatFeatureLevel(ctx.bindings().getDmlFeatureLevel()));
        } catch (DirectMlRuntimeException e) {
            closeOwnedQuietly();
            throw new EmbeddingException("Failed to upload weights to GPU for " + cfg.modelName(), e);
        } catch (RuntimeException e) {
            closeOwnedQuietly();
            throw e;
        }
    }

    @Override public boolean isReady() { return ready; }
    @Override public int dimension()   { return cfg.outputDimension(); }

    /** Underlying generic config – exposed so callers can introspect the active model. */
    public BertEncoderConfig config() { return cfg; }

    @Override
    public EmbeddingVector embed(EmbeddingRequest request) throws EmbeddingException {
        if (!ready) throw new EmbeddingException("DirectMlBertEncoder is closed");
        String text = request.prefix() != null ? request.prefix() + request.text() : request.text();
        EncoderTokenizer.Encoded encoded = tokenizer.encode(text);
        int seqLen = encoded.length();
        if (seqLen < 2) {
            throw new EmbeddingException("Tokenization produced empty sequence");
        }

        int H = cfg.hiddenSize();
        int B = bucketFor(seqLen);

        float[] x = BertEmbeddingLookup.lookup(cfg,
                weights.wordEmbeddings, weights.positionEmbeddings,
                weights.tokenTypeEmbeddings, encoded, B);

        StackEntry entry;
        try {
            entry = stackFor(B);
        } catch (DirectMlRuntimeException e) {
            throw new EmbeddingException("Failed to build DirectML encoder stack for bucket=" + B
                    + " (seqLen=" + seqLen + ")", e);
        }

        float[] mask = BertPoolingWeights.additiveMask(encoded.attentionMask(), seqLen, B);
        float[] meanWeights;
        try {
            meanWeights = BertPoolingWeights.mean(encoded.attentionMask(), seqLen, B);
        } catch (IllegalStateException e) {
            throw new EmbeddingException(e.getMessage());
        }

        float[] pooled = new float[H];
        try {
            CpuTensor xCpu = CpuTensor.float32(TensorShape.of(B, H), x);
            CpuTensor maskCpu = CpuTensor.float32(TensorShape.of(B), mask);
            CpuTensor wCpu = CpuTensor.float32(TensorShape.of(B), meanWeights);
            entry.xIn.upload(xCpu);
            entry.mask.upload(maskCpu);
            entry.meanWeights.upload(wCpu);

            DirectMlTensor xInT = tensorOf(entry.xIn, B, H);
            DirectMlTensor xOutT = tensorOf(entry.xOut, B, H);
            DirectMlTensor maskT = tensorOf(entry.mask, B);
            DirectMlTensor wT    = tensorOf(entry.meanWeights, B);
            DirectMlTensor pooledT = tensorOf(entry.pooled, H);
            DirectMlTensor embGT  = tensorOf(embLnGammaBuf, H);
            DirectMlTensor embBT  = tensorOf(embLnBetaBuf,  H);

            entry.stack.dispatch(xInT, embGT, embBT, gpuLayers, maskT, xOutT);
            entry.meanPool.dispatch(xOutT, wT, pooledT);

            GpuBuffer downloadFrom;
            if (request.normalize()) {
                DirectMlTensor normT = tensorOf(normalizedBuf, H);
                l2Kernel.dispatch(pooledT, normT, 1e-12f);
                downloadFrom = normalizedBuf;
            } else {
                downloadFrom = entry.pooled;
            }

            CpuTensor pooledCpu = emptyCpuTensor(H);
            downloadFrom.download(pooledCpu);
            FloatBuffer fv = pooledCpu.data().duplicate().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            fv.position(0);
            fv.get(pooled, 0, H);
        } catch (DirectMlRuntimeException e) {
            throw new EmbeddingException("DirectML dispatch failed for bucket=" + B
                    + " (seqLen=" + seqLen + ")", e);
        }

        return new EmbeddingVector(pooled, H, cfg.modelName(), request.normalize());
    }

    // ── helpers ─────────────────────────────────────────────────────────

    private synchronized StackEntry stackFor(int bucket) throws DirectMlRuntimeException {
        StackEntry cached = stackCache.get(bucket);
        if (cached != null) return cached;

        int H = cfg.hiddenSize();
        long hiddenBytes = (long) bucket * H * Float.BYTES;
        long maskBytes = (long) bucket * Float.BYTES;
        long pooledBytes = (long) H * Float.BYTES;

        GpuBuffer xIn = null, xOut = null, mask = null, meanWeights = null, pooled = null;
        DirectMlBertEncoderStack stack = null;
        DirectMlMeanPoolKernel meanPool = null;
        try {
            xIn  = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            xOut = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            mask = ctx.allocateBuffer(maskBytes,   GpuBuffer.BufferUsage.ACTIVATION);
            meanWeights = ctx.allocateBuffer(maskBytes, GpuBuffer.BufferUsage.ACTIVATION);
            pooled = ctx.allocateBuffer(pooledBytes, GpuBuffer.BufferUsage.ACTIVATION);
            stack = new DirectMlBertEncoderStack(
                    ctx, bucket, H,
                    cfg.numHeads(), cfg.headDim(),
                    cfg.intermediateSize(), cfg.numLayers(),
                    cfg.layerNormEps(), /* hasMask */ true);
            meanPool = new DirectMlMeanPoolKernel(ctx, bucket, H);
            StackEntry entry = new StackEntry(stack, xIn, xOut, mask, meanWeights, pooled, meanPool);
            stackCache.put(bucket, entry);
            log.info("DirectMlBertEncoder({}) stack ready for bucket S={} (cached so far: {})",
                    cfg.modelName(), bucket, stackCache.size());
            return entry;
        } catch (DirectMlRuntimeException | RuntimeException e) {
            if (meanPool != null) try { meanPool.close(); } catch (Exception ignored) {}
            if (stack != null) try { stack.close(); } catch (Exception ignored) {}
            if (pooled != null)      try { pooled.close();      } catch (Exception ignored) {}
            if (meanWeights != null) try { meanWeights.close(); } catch (Exception ignored) {}
            if (mask != null)  try { mask.close();  } catch (Exception ignored) {}
            if (xOut != null)  try { xOut.close();  } catch (Exception ignored) {}
            if (xIn != null)   try { xIn.close();   } catch (Exception ignored) {}
            throw (e instanceof DirectMlRuntimeException d) ? d
                    : new DirectMlRuntimeException("Failed to allocate stack for bucket=" + bucket, e);
        }
    }

    private GpuBuffer uploadMat(float[] data, int rows, int cols) throws DirectMlRuntimeException {
        if (data.length != (long) rows * cols) {
            throw new DirectMlRuntimeException("matrix size mismatch: " + data.length
                    + " != " + rows + "*" + cols);
        }
        CpuTensor t = CpuTensor.float32(TensorShape.of(rows, cols), data);
        GpuBuffer b = ctx.allocateBufferFor(t, GpuBuffer.BufferUsage.WEIGHT);
        b.upload(t);
        ownedGpuBuffers.add(b);
        return b;
    }

    private GpuBuffer uploadVec(float[] data, int n) throws DirectMlRuntimeException {
        if (data.length != n) {
            throw new DirectMlRuntimeException("vector size mismatch: " + data.length + " != " + n);
        }
        CpuTensor t = CpuTensor.float32(TensorShape.of(n), data);
        GpuBuffer b = ctx.allocateBufferFor(t, GpuBuffer.BufferUsage.WEIGHT);
        b.upload(t);
        ownedGpuBuffers.add(b);
        return b;
    }

    private static DirectMlTensor tensorOf(GpuBuffer buf, int... dims) {
        TensorShape s = TensorShape.of(dims);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private static CpuTensor emptyCpuTensor(int n) {
        ByteBuffer storage = ByteBuffer.allocateDirect(n * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        TensorShape s = TensorShape.of(n);
        return new CpuTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, storage);
    }

    @Override
    public synchronized void close() {
        if (!ready) return;
        ready = false;
        for (StackEntry e : stackCache.values()) e.close();
        stackCache.clear();
        if (l2Kernel != null) {
            try { l2Kernel.close(); } catch (Exception ignored) {}
        }
        closeOwnedQuietly();
        if (ownsCtx) {
            try { ctx.close(); } catch (Exception ignored) {}
        }
    }

    private void closeOwnedQuietly() {
        for (int i = ownedGpuBuffers.size() - 1; i >= 0; i--) {
            try { ownedGpuBuffers.get(i).close(); } catch (Exception ignored) {}
        }
        ownedGpuBuffers.clear();
    }

    /** Number of cached form-bound stacks (one per active pad bucket). */
    public int cachedStackCount() { return stackCache.size(); }
}

