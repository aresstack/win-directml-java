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
import com.aresstack.windirectml.runtime.kernels.DirectMlBatchedL2NormalizeKernel;
import com.aresstack.windirectml.runtime.kernels.DirectMlBatchedMeanPoolKernel;
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
    private final Map<Long, BatchStackEntry> batchStackCache = new HashMap<>();
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
     * Per-{@code (bucket, batch)} resources for the batched
     * {@link #embedBatch(List)} path. Mean-pool and L2 normalisation
     * both run on the GPU via {@link DirectMlBatchedMeanPoolKernel} and
     * {@link DirectMlBatchedL2NormalizeKernel}, so the readback shrinks
     * from {@code [N*S, H]} (encoder output stack) to {@code [N, H]}
     * (pooled / normalised vectors).
     */
    private record BatchStackEntry(int bucket, int batch,
                                   DirectMlBertEncoderStack stack,
                                   GpuBuffer xIn, GpuBuffer xOut, GpuBuffer mask,
                                   GpuBuffer meanWeights, GpuBuffer pooled, GpuBuffer normalized,
                                   DirectMlBatchedMeanPoolKernel meanPool,
                                   DirectMlBatchedL2NormalizeKernel l2,
                                   CpuTensor pooledReadback,
                                   CpuTensor normalizedReadback) implements AutoCloseable {
        @Override public void close() {
            try {
                l2.close();
            } catch (Exception ignored) {
            }
            try {
                meanPool.close();
            } catch (Exception ignored) {
            }
            try {
                stack.close();
            } catch (Exception ignored) {
            }
            try {
                xIn.close();
            } catch (Exception ignored) {
            }
            try {
                xOut.close();
            } catch (Exception ignored) {
            }
            try {
                mask.close();
            } catch (Exception ignored) {
            }
            try {
                meanWeights.close();
            } catch (Exception ignored) {
            }
            try {
                pooled.close();
            } catch (Exception ignored) {
            }
            try {
                normalized.close();
            } catch (Exception ignored) {
            }
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

    /**
     * Bucket-batched embedding path: groups requests by pad-bucket and
     * runs <strong>one</strong> DirectML encoder forward + one batched
     * mean-pool + (optionally) one batched L2-normalise per bucket on a
     * {@code [N, S, H]} stack instead of {@code N} separate forwards.
     * The PCIe read-back shrinks to {@code [N, H]} per dispatched bucket
     * (plus a second {@code [N, H]} download only for buckets with
     * mixed {@code normalize} flags).
     */
    @Override
    public List<EmbeddingVector> embedBatch(List<EmbeddingRequest> requests) throws EmbeddingException {
        Objects.requireNonNull(requests, "requests");
        if (requests.isEmpty()) {
            throw new IllegalArgumentException("requests must not be empty");
        }
        if (!ready) throw new EmbeddingException("DirectMlBertEncoder is closed");

        int n = requests.size();
        int H = cfg.hiddenSize();
        EmbeddingVector[] out = new EmbeddingVector[n];

        // 1. Tokenize + group by pad-bucket.
        EncoderTokenizer.Encoded[] encs = new EncoderTokenizer.Encoded[n];
        Map<Integer, List<Integer>> byBucket = new HashMap<>();
        for (int i = 0; i < n; i++) {
            EmbeddingRequest r = requests.get(i);
            String text = r.prefix() != null ? r.prefix() + r.text() : r.text();
            EncoderTokenizer.Encoded enc = tokenizer.encode(text);
            if (enc.length() < 2) {
                throw new EmbeddingException("Tokenization produced empty sequence at index " + i);
            }
            encs[i] = enc;
            int b = bucketFor(enc.length());
            byBucket.computeIfAbsent(b, k -> new ArrayList<>()).add(i);
        }

        // 2. One batched dispatch per bucket.
        for (Map.Entry<Integer, List<Integer>> e : byBucket.entrySet()) {
            int B = e.getKey();
            List<Integer> group = e.getValue();
            int N = group.size();

            float[] xBatch = new float[N * B * H];
            float[] maskBatch = new float[N * B];
            float[] meanWeightsBatch = new float[N * B];
            boolean anyNormalize = false;
            boolean allNormalize = true;
            for (int gi = 0; gi < N; gi++) {
                int i = group.get(gi);
                EncoderTokenizer.Encoded enc = encs[i];
                float[] xi = BertEmbeddingLookup.lookup(cfg,
                        weights.wordEmbeddings, weights.positionEmbeddings,
                        weights.tokenTypeEmbeddings, enc, B);
                System.arraycopy(xi, 0, xBatch, gi * B * H, B * H);
                float[] mi = BertPoolingWeights.additiveMask(enc.attentionMask(), enc.length(), B);
                System.arraycopy(mi, 0, maskBatch, gi * B, B);
                float[] wi;
                try {
                    wi = BertPoolingWeights.mean(enc.attentionMask(), enc.length(), B);
                } catch (IllegalStateException ise) {
                    throw new EmbeddingException(ise.getMessage());
                }
                System.arraycopy(wi, 0, meanWeightsBatch, gi * B, B);
                boolean norm = requests.get(i).normalize();
                anyNormalize |= norm;
                allNormalize &= norm;
            }

            BatchStackEntry entry;
            try {
                entry = batchStackFor(B, N);
            } catch (DirectMlRuntimeException de) {
                throw new EmbeddingException("Failed to build batched DirectML encoder stack for bucket="
                        + B + ", batch=" + N, de);
            }

            try {
                CpuTensor xCpu = CpuTensor.float32(TensorShape.of(N * B, H), xBatch);
                CpuTensor mCpu = CpuTensor.float32(TensorShape.of(N * B), maskBatch);
                CpuTensor wCpu = CpuTensor.float32(TensorShape.of(N, B), meanWeightsBatch);
                entry.xIn.upload(xCpu);
                entry.mask.upload(mCpu);
                entry.meanWeights.upload(wCpu);

                DirectMlTensor xInT = tensorOf(entry.xIn, N * B, H);
                DirectMlTensor xOutT = tensorOf(entry.xOut, N * B, H);
                DirectMlTensor maskT = tensorOf(entry.mask, N * B);
                DirectMlTensor wT = tensorOf(entry.meanWeights, N, B);
                DirectMlTensor pooledT = tensorOf(entry.pooled, N, H);
                DirectMlTensor embGT = tensorOf(embLnGammaBuf, H);
                DirectMlTensor embBT = tensorOf(embLnBetaBuf, H);

                // GPU: encoder stack → batched mean-pool [N,H].
                entry.stack.dispatch(xInT, embGT, embBT, gpuLayers, maskT, xOutT);
                entry.meanPool.dispatch(xOutT, wT, pooledT);

                // GPU: batched L2 only if any row in this bucket asks for it.
                if (anyNormalize) {
                    DirectMlTensor normT = tensorOf(entry.normalized, N, H);
                    entry.l2.dispatch(pooledT, normT, 1e-12f);
                }

                // Read back only the [N, H] vectors we actually need.
                FloatBuffer pooledFv = null;
                if (!allNormalize) {
                    entry.pooled.download(entry.pooledReadback);
                    pooledFv = entry.pooledReadback.data().duplicate()
                            .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                }
                FloatBuffer normFv = null;
                if (anyNormalize) {
                    entry.normalized.download(entry.normalizedReadback);
                    normFv = entry.normalizedReadback.data().duplicate()
                            .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                }

                for (int gi = 0; gi < N; gi++) {
                    int i = group.get(gi);
                    boolean normalize = requests.get(i).normalize();
                    float[] vec = new float[H];
                    FloatBuffer src = normalize ? normFv : pooledFv;
                    src.position(gi * H);
                    src.get(vec, 0, H);
                    out[i] = new EmbeddingVector(vec, H, cfg.modelName(), normalize);
                }
            } catch (DirectMlRuntimeException de) {
                throw new EmbeddingException("Batched DirectML dispatch failed for bucket="
                        + B + ", batch=" + N, de);
            }
        }

        return List.of(out);
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

    private synchronized BatchStackEntry batchStackFor(int bucket, int batch) throws DirectMlRuntimeException {
        long key = ((long) bucket << 32) | ((long) batch & 0xffffffffL);
        BatchStackEntry cached = batchStackCache.get(key);
        if (cached != null) return cached;

        int H = cfg.hiddenSize();
        long rowsBytes = (long) batch * bucket * H * Float.BYTES;
        long maskBytes = (long) batch * bucket * Float.BYTES;
        long pooledBytes = (long) batch * H * Float.BYTES;

        GpuBuffer xIn = null, xOut = null, mask = null;
        GpuBuffer meanWeights = null, pooled = null, normalized = null;
        DirectMlBertEncoderStack stack = null;
        DirectMlBatchedMeanPoolKernel meanPool = null;
        DirectMlBatchedL2NormalizeKernel l2 = null;
        try {
            xIn = ctx.allocateBuffer(rowsBytes, GpuBuffer.BufferUsage.ACTIVATION);
            xOut = ctx.allocateBuffer(rowsBytes, GpuBuffer.BufferUsage.ACTIVATION);
            mask = ctx.allocateBuffer(maskBytes, GpuBuffer.BufferUsage.ACTIVATION);
            meanWeights = ctx.allocateBuffer(maskBytes, GpuBuffer.BufferUsage.ACTIVATION);
            pooled = ctx.allocateBuffer(pooledBytes, GpuBuffer.BufferUsage.ACTIVATION);
            normalized = ctx.allocateBuffer(pooledBytes, GpuBuffer.BufferUsage.ACTIVATION);
            stack = new DirectMlBertEncoderStack(
                    ctx, batch, bucket, H,
                    cfg.numHeads(), cfg.headDim(),
                    cfg.intermediateSize(), cfg.numLayers(),
                    cfg.layerNormEps(), /* hasMask */ true);
            meanPool = new DirectMlBatchedMeanPoolKernel(ctx, batch, bucket, H);
            l2 = new DirectMlBatchedL2NormalizeKernel(ctx, batch, H, 1e-12f);
            CpuTensor pooledReadback = emptyCpuTensor(batch * H);
            CpuTensor normalizedReadback = emptyCpuTensor(batch * H);
            BatchStackEntry entry = new BatchStackEntry(bucket, batch, stack,
                    xIn, xOut, mask, meanWeights, pooled, normalized,
                    meanPool, l2, pooledReadback, normalizedReadback);
            batchStackCache.put(key, entry);
            log.info("DirectMlBertEncoder({}) batched stack ready for bucket S={}, batch N={} (cached so far: {})",
                    cfg.modelName(), bucket, batch, batchStackCache.size());
            return entry;
        } catch (DirectMlRuntimeException | RuntimeException ex) {
            if (l2 != null) try {
                l2.close();
            } catch (Exception ignored) {
            }
            if (meanPool != null) try {
                meanPool.close();
            } catch (Exception ignored) {
            }
            if (stack != null) try {
                stack.close();
            } catch (Exception ignored) {
            }
            if (normalized != null) try {
                normalized.close();
            } catch (Exception ignored) {
            }
            if (pooled != null) try {
                pooled.close();
            } catch (Exception ignored) {
            }
            if (meanWeights != null) try {
                meanWeights.close();
            } catch (Exception ignored) {
            }
            if (mask != null) try {
                mask.close();
            } catch (Exception ignored) {
            }
            if (xOut != null) try {
                xOut.close();
            } catch (Exception ignored) {
            }
            if (xIn != null) try {
                xIn.close();
            } catch (Exception ignored) {
            }
            throw (ex instanceof DirectMlRuntimeException d) ? d
                    : new DirectMlRuntimeException("Failed to allocate batched stack for bucket="
                            + bucket + ", batch=" + batch, ex);
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
        for (BatchStackEntry e : batchStackCache.values()) e.close();
        batchStackCache.clear();
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

    /** Number of cached batched stacks (one per active {@code (bucket, batch)} pair). */
    public int cachedBatchStackCount() { return batchStackCache.size(); }
}

