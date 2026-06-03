package com.aresstack.windirectml.encoder.minilm;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.encoder.EncoderTokenizer;
import com.aresstack.windirectml.encoder.bert.BertEmbeddingLookup;
import com.aresstack.windirectml.encoder.bert.BertEncoderConfig;
import com.aresstack.windirectml.encoder.bert.BertGpuLayerWeights;
import com.aresstack.windirectml.encoder.bert.BertPoolingWeights;
import com.aresstack.windirectml.encoder.bert.DirectMlBertEncoderStack;
import com.aresstack.windirectml.encoder.tokenizer.WordPieceTokenizer;
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
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * MiniLM-specific {@link EmbeddingModel} that drives the generic
 * BERT-style DirectML encoder pipeline
 * ({@link DirectMlBertEncoderStack} + {@link BertGpuLayerWeights}) with
 * the {@code sentence-transformers/all-MiniLM-L6-v2} weights/tokenizer.
 * <p>
 * Pipeline:
 * <pre>
 *   text
 *     → WordPieceTokenizer
 *     → BertEmbeddingLookup (CPU word + position + tokenType sum)
 *     → Upload to GPU
 *     → DirectMlBertEncoderStack (Embedding-LN + N Layer)
 *     → DirectMlMeanPoolKernel (GEMM, w[t]=m[t]/Σm pre-normalised on CPU)
 *     → DirectMlL2NormalizeKernel (GEMM-square-sum + SQRT(ε² bias) + DIVIDE-broadcast)
 *     → Download [H]
 *     → EmbeddingVector
 * </pre>
 * Pooling and L2-normalisation both run on the GPU; only the final
 * {@code H} floats are read back per inference. All DirectML primitives
 * (GEMM, ELEMENT_WISE_SQRT, ELEMENT_WISE_DIVIDE) are FL-1.0 baseline.
 * <p>
 * The DirectML stacks are form-bound on {@code seq}. Inputs are padded
 * to a bucket {@code S ∈ {64, 128, 256, 512}}; the stack cache therefore
 * holds at most four entries per encoder. Padded positions are
 * neutralised in attention via a {@code -1e9} additive mask and ignored
 * in MeanPool/L2 via the original {@code attentionMask}.
 * <p>
 * The MiniLM-specific class names ({@code MiniLmConfig},
 * {@code CpuMiniLmWeights}) survive only at the model-loading boundary;
 * the GPU dispatch path consumes the model-agnostic
 * {@link BertEncoderConfig} and {@link BertGpuLayerWeights}. Adding a
 * new BERT-style family (E5, JinaBERT) reuses the same generic stack.
 */
public final class DirectMlMiniLmEncoder implements EmbeddingModel, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlMiniLmEncoder.class);

    /**
     * Pad-buckets used to coalesce arbitrary tokenizer sequence lengths
     * onto a small set of fixed DirectML stack shapes. Ordered ascending.
     * MiniLM's {@code maxPositionEmbeddings} is 512; any seqLen above the
     * largest bucket falls through to an exact-length stack (unbucketed).
     */
    private static final int[] PAD_BUCKETS = {64, 128, 256, 512};

    /**
     * Defensive-copy accessor for the configured pad-buckets.
     */
    public static int[] buckets() {
        return PAD_BUCKETS.clone();
    }

    /**
     * Smallest bucket {@code b} such that {@code b >= seqLen}, else {@code seqLen}.
     */
    public static int bucketFor(int seqLen) {
        for (int b : PAD_BUCKETS) {
            if (b >= seqLen) return b;
        }
        return seqLen;
    }

    private final MiniLmArchitecture architecture;
    private final BertEncoderConfig cfg;
    private final CpuMiniLmWeights weights;
    private final EncoderTokenizer tokenizer;
    private final DirectMlContextImpl ctx;
    private final boolean ownsCtx;

    // GPU-resident weights (1× upload, life-of-encoder lifetime).
    private final List<GpuBuffer> ownedGpuBuffers = new ArrayList<>();
    private final GpuBuffer embLnGammaBuf;
    private final GpuBuffer embLnBetaBuf;
    private final List<BertGpuLayerWeights> gpuLayers;

    private final DirectMlL2NormalizeKernel l2Kernel;
    private final GpuBuffer normalizedBuf;

    private final Map<Integer, StackEntry> stackCache = new HashMap<>();
    private volatile boolean ready = false;

    private record StackEntry(DirectMlBertEncoderStack stack,
                              GpuBuffer xIn,
                              GpuBuffer xOut,
                              GpuBuffer mask,
                              GpuBuffer meanWeights,
                              GpuBuffer pooled,
                              DirectMlMeanPoolKernel meanPool) implements AutoCloseable {
        @Override
        public void close() {
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
        }
    }

    /**
     * Construct an encoder that owns its own {@link DirectMlContextImpl}.
     */
    public static DirectMlMiniLmEncoder load(Path modelDir) throws EmbeddingException {
        return load(modelDir, "directml");
    }

    /**
     * Construct an encoder that owns its own {@link DirectMlContextImpl}.
     */
    public static DirectMlMiniLmEncoder load(Path modelDir, String nativeBackend) throws EmbeddingException {
        DirectMlContextImpl ctx = null;
        try {
            if (!WindowsBindings.isSupported()) {
                throw new EmbeddingException("DirectML requires Windows + D3D12 on this host");
            }
            ctx = new DirectMlContextImpl(normalizeNativeBackend(nativeBackend));
            ctx.initialize();
            if (!ctx.isReady() || !ctx.bindings().hasDirectMl()) {
                throw new EmbeddingException("No DirectML device available on this adapter");
            }
            int fl = ctx.bindings().getDmlFeatureLevel();
            if (!DirectMlBindings.supportsFusedGelu(fl)) {
                org.slf4j.LoggerFactory.getLogger(DirectMlMiniLmEncoder.class)
                        .info("DirectMlMiniLmEncoder: FL={} – using composite GELU fallback",
                                DirectMlBindings.formatFeatureLevel(fl));
            }

            MiniLmArchitecture arch = new MiniLmArchitecture();
            CpuMiniLmWeights w = CpuMiniLmWeights.load(modelDir, arch);
            WordPieceTokenizer t = WordPieceTokenizer.load(modelDir.resolve("tokenizer.json"),
                    arch.config().maxPositionEmbeddings());
            return new DirectMlMiniLmEncoder(ctx, /* ownsCtx */ true, arch, w, t);
        } catch (EmbeddingException e) {
            if (ctx != null) try {
                ctx.close();
            } catch (Exception ignored) {
            }
            throw e;
        } catch (Exception e) {
            if (ctx != null) try {
                ctx.close();
            } catch (Exception ignored) {
            }
            throw new EmbeddingException("Failed to load DirectMlMiniLmEncoder from " + modelDir, e);
        }
    }

    private static String normalizeNativeBackend(String nativeBackend) {
        if (nativeBackend == null || nativeBackend.trim().isEmpty()) {
            return "directml";
        }
        return nativeBackend.trim();
    }

    public DirectMlMiniLmEncoder(DirectMlContextImpl ctx,
                                 boolean ownsCtx,
                                 MiniLmArchitecture architecture,
                                 CpuMiniLmWeights weights,
                                 EncoderTokenizer tokenizer) throws EmbeddingException {
        this.ctx = Objects.requireNonNull(ctx);
        this.ownsCtx = ownsCtx;
        this.architecture = Objects.requireNonNull(architecture);
        this.weights = Objects.requireNonNull(weights);
        this.tokenizer = Objects.requireNonNull(tokenizer);
        this.cfg = MiniLmConfigs.toBertConfig(architecture.config());
        this.cfg.validate();

        try {
            int H = cfg.hiddenSize();
            int I = cfg.intermediateSize();

            // ── Upload Embedding-LN parameters ──────────────────────────
            embLnGammaBuf = uploadVec(weights.embLnGamma, H);
            embLnBetaBuf = uploadVec(weights.embLnBeta, H);

            // ── Upload per-layer weights → generic BertGpuLayerWeights ──
            List<BertGpuLayerWeights> built = new ArrayList<>(weights.layers.size());
            for (CpuMiniLmWeights.LayerWeights l : weights.layers) {
                built.add(new BertGpuLayerWeights(
                        uploadMat(l.qWeight, H, H), uploadVec(l.qBias, H),
                        uploadMat(l.kWeight, H, H), uploadVec(l.kBias, H),
                        uploadMat(l.vWeight, H, H), uploadVec(l.vBias, H),
                        uploadMat(l.attnOutWeight, H, H), uploadVec(l.attnOutBias, H),
                        uploadVec(l.attnLnGamma, H), uploadVec(l.attnLnBeta, H),
                        uploadMat(l.mlpInterWeight, I, H), uploadVec(l.mlpInterBias, I),
                        uploadMat(l.mlpOutWeight, H, I), uploadVec(l.mlpOutBias, H),
                        uploadVec(l.outLnGamma, H), uploadVec(l.outLnBeta, H)));
            }
            this.gpuLayers = List.copyOf(built);

            // ── Shape-bound L2-normalize kernel + H-Float output buffer ─
            DirectMlL2NormalizeKernel builtL2 = null;
            GpuBuffer builtNorm = null;
            try {
                builtL2 = new DirectMlL2NormalizeKernel(ctx, H, 1e-12f);
                builtNorm = ctx.allocateBuffer((long) H * Float.BYTES,
                        GpuBuffer.BufferUsage.ACTIVATION);
            } catch (RuntimeException e) {
                if (builtL2 != null) try {
                    builtL2.close();
                } catch (Exception ignored) {
                }
                if (builtNorm != null) try {
                    builtNorm.close();
                } catch (Exception ignored) {
                }
                throw e;
            }
            this.l2Kernel = builtL2;
            this.normalizedBuf = builtNorm;
            ownedGpuBuffers.add(normalizedBuf);

            this.ready = true;
            log.info("DirectMlMiniLmEncoder ready: layers={}, hidden={}, heads={}, inter={}, FL={}",
                    cfg.numLayers(), cfg.hiddenSize(),
                    cfg.numHeads(), cfg.intermediateSize(),
                    DirectMlBindings.formatFeatureLevel(ctx.bindings().getDmlFeatureLevel()));
        } catch (DirectMlRuntimeException e) {
            closeOwnedQuietly();
            throw new EmbeddingException("Failed to upload MiniLM weights to GPU", e);
        } catch (RuntimeException e) {
            closeOwnedQuietly();
            throw e;
        }
    }

    @Override
    public boolean isReady() {
        return ready;
    }

    @Override
    public int dimension() {
        return cfg.outputDimension();
    }

    @Override
    public EmbeddingVector embed(EmbeddingRequest request) throws EmbeddingException {
        if (!ready) throw new EmbeddingException("DirectMlMiniLmEncoder is closed");
        String text = request.prefix() != null ? request.prefix() + request.text() : request.text();
        EncoderTokenizer.Encoded encoded = tokenizer.encode(text);
        int seqLen = encoded.length();
        if (seqLen < 2) {
            throw new EmbeddingException("Tokenization produced empty sequence");
        }

        int H = cfg.hiddenSize();
        int B = bucketFor(seqLen);

        // 1. CPU embedding lookup → packed [B*H] (generic helper).
        float[] x = BertEmbeddingLookup.lookup(cfg,
                weights.wordEmbeddings, weights.positionEmbeddings,
                weights.tokenTypeEmbeddings, encoded, B);

        // 2. Build/get cached generic encoder stack and scratch buffers.
        StackEntry entry;
        try {
            entry = stackFor(B);
        } catch (DirectMlRuntimeException e) {
            throw new EmbeddingException("Failed to build DirectML encoder stack for bucket=" + B
                    + " (seqLen=" + seqLen + ")", e);
        }

        // 3. Additive attention mask + pre-normalised mean-pool weights.
        float[] mask = BertPoolingWeights.additiveMask(encoded.attentionMask(), seqLen, B);
        float[] meanWeights;
        try {
            meanWeights = BertPoolingWeights.mean(encoded.attentionMask(), seqLen, B);
        } catch (IllegalStateException e) {
            throw new EmbeddingException(e.getMessage());
        }

        // 4. Upload + dispatch (encoder stack → mean-pool → optional L2).
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
            DirectMlTensor wT = tensorOf(entry.meanWeights, B);
            DirectMlTensor pooledT = tensorOf(entry.pooled, H);
            DirectMlTensor embGT = tensorOf(embLnGammaBuf, H);
            DirectMlTensor embBT = tensorOf(embLnBetaBuf, H);

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

        return new EmbeddingVector(pooled, H, MiniLmArchitecture.NAME, request.normalize());
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
            xIn = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            xOut = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            mask = ctx.allocateBuffer(maskBytes, GpuBuffer.BufferUsage.ACTIVATION);
            meanWeights = ctx.allocateBuffer(maskBytes, GpuBuffer.BufferUsage.ACTIVATION);
            pooled = ctx.allocateBuffer(pooledBytes, GpuBuffer.BufferUsage.ACTIVATION);
            stack = new DirectMlBertEncoderStack(
                    ctx, bucket, H,
                    cfg.numHeads(), cfg.headDim(),
                    cfg.intermediateSize(), cfg.numLayers(),
                    cfg.layerNormEps(),     /* hasMask */ true);
            meanPool = new DirectMlMeanPoolKernel(ctx, bucket, H);
            StackEntry entry = new StackEntry(stack, xIn, xOut, mask, meanWeights, pooled, meanPool);
            stackCache.put(bucket, entry);
            log.info("DirectMlMiniLmEncoder stack ready for bucket S={} (cached buckets so far: {})",
                    bucket, stackCache.size());
            return entry;
        } catch (RuntimeException e) {
            if (meanPool != null) try {
                meanPool.close();
            } catch (Exception ignored) {
            }
            if (stack != null) try {
                stack.close();
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
            try {
                l2Kernel.close();
            } catch (Exception ignored) {
            }
        }
        closeOwnedQuietly();
        if (ownsCtx) {
            try {
                ctx.close();
            } catch (Exception ignored) {
            }
        }
    }

    private void closeOwnedQuietly() {
        for (int i = ownedGpuBuffers.size() - 1; i >= 0; i--) {
            try {
                ownedGpuBuffers.get(i).close();
            } catch (Exception ignored) {
            }
        }
        ownedGpuBuffers.clear();
    }

    /**
     * Number of cached form-bound stacks (one per active pad bucket).
     */
    public int cachedStackCount() {
        return stackCache.size();
    }
}

