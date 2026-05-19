package com.aresstack.windirectml.encoder.minilm;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.encoder.EncoderTokenizer;
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
 * Vollständige DirectML-Variante von {@code all-MiniLM-L6-v2}.
 * <p>
 * Pipeline:
 * <pre>
 *   text
 *     → WordPieceTokenizer
 *     → CPU Embedding-Lookup (word + position + tokenType)
 *     → Upload nach GPU
 *     → DirectMlMiniLmEncoderStack (Embedding-LN + N Layer)
 *     → DirectMlMeanPoolKernel (GEMM, w[t]=m[t]/Σm vorab CPU-normiert)
 *     → DirectMlL2NormalizeKernel (GEMM-square-sum + SQRT(scale=1, bias=ε²) + DIVIDE-broadcast)
 *     → Download [H]
 *     → EmbeddingVector
 * </pre>
 * Pooling und L2-Normalisierung laufen damit beide auf der GPU; pro
 * Inferenz werden nur noch {@code H} statt {@code B·H} Floats über PCIe
 * gelesen, und der finale Vektor ist bereits normiert. Alle eingesetzten
 * DirectML-Primitive (GEMM, ELEMENT_WISE_SQRT, ELEMENT_WISE_DIVIDE) sind
 * FL-1.0-Baseline und in jeder ausgelieferten {@code DirectML.dll}
 * (inkl. Windows 11 In-Box 1.8.0) vorhanden.
 * Die DirectML-Stacks sind formgebunden auf {@code seq}. Eingaben werden
 * auf einen Pad-Bucket {@code S ∈ {64, 128, 256, 512}} aufgefüllt; der
 * Stack-Cache hält damit höchstens vier Einträge pro Encoder
 * (statt einem pro tatsächlich vorgekommener Sequenz-Länge). Padded
 * Positionen werden in der Attention durch eine Mask von {@code -1e9}
 * deaktiviert und in MeanPool/L2 durch die ursprüngliche
 * {@code attentionMask} ignoriert.
 * <p>
 * Alle 16 Gewichts-Buffer pro Layer sowie das Embedding-LN-Paar werden
 * im Konstruktor einmal hochgeladen und für die Lebensdauer des Encoders
 * gehalten.
 * <p>
 * <b>Feature levels:</b> {@code DirectMlMiniLmLayerBlock} bezieht GELU
 * über {@code GeluKernel.create(ctx, n)}. Auf
 * {@code DML_FEATURE_LEVEL_5_1} und höher wird die native fused GELU
 * ({@code DML_OPERATOR_ACTIVATION_GELU}) verwendet, auf älteren
 * Feature-Levels (z. B. Windows-11-In-Box DirectML 1.8.0, FL 5.0) wird
 * automatisch der Composite-Fallback (ERF + IDENTITY + MULTIPLY, alle
 * FL 2.0) gewählt. Der Encoder läuft damit auf jeder ausgelieferten
 * {@code DirectML.dll} ohne Redist-Pflicht.
 */
public final class DirectMlMiniLmEncoder implements EmbeddingModel, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlMiniLmEncoder.class);

    /**
     * Pad-buckets used to coalesce arbitrary tokenizer sequence lengths
     * onto a small set of fixed DirectML stack shapes. Ordered ascending.
     * MiniLM's {@code maxPositionEmbeddings} is 512; any seqLen above the
     * largest bucket falls through to an exact-length stack (unbucketed).
     * Kept private to prevent external mutation of the array contents;
     * use {@link #buckets()} for read-only access.
     */
    private static final int[] PAD_BUCKETS = {64, 128, 256, 512};

    /**
     * Returns a defensive copy of the configured pad-buckets. Callers may
     * mutate the returned array without affecting the encoder.
     */
    public static int[] buckets() {
        return PAD_BUCKETS.clone();
    }

    /**
     * Selects the smallest bucket {@code b} such that {@code b >= seqLen}.
     * For {@code seqLen > 512} returns the exact length so the encoder
     * still works (no bucketing benefit in that overflow case).
     */
    public static int bucketFor(int seqLen) {
        for (int b : PAD_BUCKETS) {
            if (b >= seqLen) return b;
        }
        return seqLen;
    }

    private final MiniLmArchitecture architecture;
    private final MiniLmConfig config;
    private final CpuMiniLmWeights weights;          // Embedding-Tables + LN-Params bleiben CPU-resident
    private final EncoderTokenizer tokenizer;
    private final DirectMlContextImpl ctx;
    private final boolean ownsCtx;

    // GPU-resident weights (1× upload, life-of-encoder lifetime).
    private final List<GpuBuffer> ownedGpuBuffers = new ArrayList<>();
    private final GpuBuffer embLnGammaBuf;
    private final GpuBuffer embLnBetaBuf;
    private final List<DirectMlMiniLmLayerBlock.LayerWeights> gpuLayers;

    /**
     * Shape-bound L2-normalize kernel ({@code N = hiddenSize},
     * {@code ε = 1e-12f}). Shared across all pad-buckets since L2 only
     * acts on the {@code [H]} pooled vector. {@code null} → not built yet
     * (constructor failed before this line, see error path).
     */
    private final DirectMlL2NormalizeKernel l2Kernel;

    /**
     * H-Float GPU output buffer for the L2-normalised vector. Re-used
     * across every {@code embed()} call; downloaded once per inference.
     */
    private final GpuBuffer normalizedBuf;

    // Per-bucket stack cache. Stacks are form-bound, so each bucket
    // size triggers exactly one entry; weights are shared across entries.
    private final Map<Integer, StackEntry> stackCache = new HashMap<>();

    private volatile boolean ready = false;

    private record StackEntry(DirectMlMiniLmEncoderStack stack,
                              GpuBuffer xIn,
                              GpuBuffer xOut,
                              GpuBuffer mask,
                              GpuBuffer meanWeights,
                              GpuBuffer pooled,
                              DirectMlMeanPoolKernel meanPool) implements AutoCloseable {
        @Override public void close() {
            try { meanPool.close();   } catch (Exception ignored) {}
            try { stack.close();      } catch (Exception ignored) {}
            try { xIn.close();        } catch (Exception ignored) {}
            try { xOut.close();       } catch (Exception ignored) {}
            try { mask.close();       } catch (Exception ignored) {}
            try { meanWeights.close();} catch (Exception ignored) {}
            try { pooled.close();     } catch (Exception ignored) {}
        }
    }

    /**
     * Construct an encoder that owns its own {@link DirectMlContextImpl}.
     * The context is initialised eagerly. Uses the native fused GELU on
     * {@code DML_FEATURE_LEVEL_5_1} and falls back to the composite
     * ERF+IDENTITY+MULTIPLY GELU on older DirectML feature levels, so the
     * encoder runs on any shipping Windows-10/11 in-box {@code DirectML.dll}.
     */
    public static DirectMlMiniLmEncoder load(Path modelDir) throws EmbeddingException {
        DirectMlContextImpl ctx = null;
        try {
            if (!WindowsBindings.isSupported()) {
                throw new EmbeddingException("DirectML requires Windows + D3D12 on this host");
            }
            ctx = new DirectMlContextImpl("directml");
            ctx.initialize();
            if (!ctx.isReady() || !ctx.bindings().hasDirectMl()) {
                throw new EmbeddingException("No DirectML device available on this adapter");
            }
            int fl = ctx.bindings().getDmlFeatureLevel();
            // DirectMlMiniLmLayerBlock uses GeluKernel.create(...) which selects
            // the native fused GELU on FL>=5.1 and the ERF+IDENTITY+MULTIPLY
            // composite fallback on FL<5.1 — so the whole pipeline runs on
            // every shipping Windows-10/11 in-box DirectML.dll. We just log
            // the chosen strategy here.
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
            if (ctx != null) try { ctx.close(); } catch (Exception ignored) {}
            throw e;
        } catch (Exception e) {
            if (ctx != null) try { ctx.close(); } catch (Exception ignored) {}
            throw new EmbeddingException("Failed to load DirectMlMiniLmEncoder from " + modelDir, e);
        }
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
        this.config = architecture.config();

        try {
            // ── Upload Embedding-LN-Parameter ──────────────────────────
            embLnGammaBuf = uploadVec(weights.embLnGamma, config.hiddenSize());
            embLnBetaBuf  = uploadVec(weights.embLnBeta,  config.hiddenSize());

            // ── Upload pro Layer alle 16 Gewichts-Tensoren ─────────────
            List<DirectMlMiniLmLayerBlock.LayerWeights> built = new ArrayList<>(weights.layers.size());
            int H = config.hiddenSize();
            int I = config.intermediateSize();
            for (CpuMiniLmWeights.LayerWeights l : weights.layers) {
                GpuBuffer qw = uploadMat(l.qWeight, H, H);
                GpuBuffer qb = uploadVec(l.qBias,  H);
                GpuBuffer kw = uploadMat(l.kWeight, H, H);
                GpuBuffer kb = uploadVec(l.kBias,  H);
                GpuBuffer vw = uploadMat(l.vWeight, H, H);
                GpuBuffer vb = uploadVec(l.vBias,  H);
                GpuBuffer ow = uploadMat(l.attnOutWeight, H, H);
                GpuBuffer ob = uploadVec(l.attnOutBias,   H);
                GpuBuffer ag = uploadVec(l.attnLnGamma,   H);
                GpuBuffer ab = uploadVec(l.attnLnBeta,    H);
                GpuBuffer iw = uploadMat(l.mlpInterWeight, I, H);
                GpuBuffer ib = uploadVec(l.mlpInterBias,   I);
                GpuBuffer mw = uploadMat(l.mlpOutWeight,   H, I);
                GpuBuffer mb = uploadVec(l.mlpOutBias,     H);
                GpuBuffer og = uploadVec(l.outLnGamma, H);
                GpuBuffer ob2 = uploadVec(l.outLnBeta,  H);
                built.add(new DirectMlMiniLmLayerBlock.LayerWeights(
                        qw, qb, kw, kb, vw, vb, ow, ob, ag, ab,
                        iw, ib, mw, mb, og, ob2));
            }
            this.gpuLayers = List.copyOf(built);

            // ── Shape-bound L2-normalize kernel + H-Float output buffer ─
            // The L2 path is the same for every pad-bucket because it
            // only operates on the [H] pooled vector. Built once here so
            // the dispatch tail in embed() does not pay a per-call op
            // construction cost. ε is baked into the SQRT-ScaleBias.
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
            log.info("DirectMlMiniLmEncoder ready: layers={}, hidden={}, heads={}, inter={}, FL={}",
                    config.numLayers(), config.hiddenSize(),
                    config.numAttentionHeads(), config.intermediateSize(),
                    DirectMlBindings.formatFeatureLevel(ctx.bindings().getDmlFeatureLevel()));
        } catch (DirectMlRuntimeException e) {
            closeOwnedQuietly();
            throw new EmbeddingException("Failed to upload MiniLM weights to GPU", e);
        } catch (RuntimeException e) {
            closeOwnedQuietly();
            throw e;
        }
    }

    @Override public boolean isReady() { return ready; }
    @Override public int dimension()   { return architecture.outputDimension(); }

    @Override
    public EmbeddingVector embed(EmbeddingRequest request) throws EmbeddingException {
        if (!ready) throw new EmbeddingException("DirectMlMiniLmEncoder is closed");
        String text = request.prefix() != null ? request.prefix() + request.text() : request.text();
        EncoderTokenizer.Encoded encoded = tokenizer.encode(text);
        int seqLen = encoded.length();
        if (seqLen < 2) {
            throw new EmbeddingException("Tokenization produced empty sequence");
        }

        int H = config.hiddenSize();
        // Pad-bucket: pick smallest bucket >= seqLen so we reuse one stack
        // per bucket (S∈{64,128,256,512}) instead of one per actual length.
        int B = bucketFor(seqLen);

        // 1. CPU embedding lookup (word + pos + tokenType) for the first
        // seqLen rows; rows [seqLen, B) stay zero. Padded rows never feed
        // back into valid rows (attention mask + per-row LN/Linear), and
        // MeanPool reads only valid rows via the original attentionMask.
        float[] x = new float[B * H];
        for (int t = 0; t < seqLen; t++) {
            int id  = encoded.inputIds()[t];
            int tt  = encoded.tokenTypeIds()[t];
            int dst = t * H;
            int ws  = id * H;
            int ps  = t * H;
            int tts = tt * H;
            for (int h = 0; h < H; h++) {
                x[dst + h] = weights.wordEmbeddings[ws + h]
                        + weights.positionEmbeddings[ps + h]
                        + weights.tokenTypeEmbeddings[tts + h];
            }
        }

        // 2. Build/get cached stack and per-bucket scratch buffers.
        StackEntry entry;
        try {
            entry = stackFor(B);
        } catch (DirectMlRuntimeException e) {
            throw new EmbeddingException("Failed to build DirectML encoder stack for bucket=" + B
                    + " (seqLen=" + seqLen + ")", e);
        }

        // 3. Mask: 0.0 valid (incl. all real tokens, attentionMask=1),
        //    -1e9 for padded bucket positions [seqLen, B).
        float[] mask = new float[B];
        for (int i = 0; i < seqLen; i++) {
            mask[i] = encoded.attentionMask()[i] == 0 ? -1e9f : 0f;
        }
        for (int i = seqLen; i < B; i++) {
            mask[i] = -1e9f;
        }

        // 3b. Pre-normalised mean-pool weights on CPU: w[t] = m[t] / Σm
        // for the valid prefix [0, seqLen); padded positions stay 0.
        // This collapses the GPU pooling to a single GEMM (FL 1.0) and
        // removes the need for a reduce/divide on the device.
        int validCount = 0;
        for (int i = 0; i < seqLen; i++) {
            if (encoded.attentionMask()[i] != 0) validCount++;
        }
        if (validCount == 0) {
            throw new EmbeddingException("Attention mask is all zero – nothing to pool");
        }
        float invValid = 1.0f / validCount;
        float[] meanWeights = new float[B];
        for (int i = 0; i < seqLen; i++) {
            if (encoded.attentionMask()[i] != 0) meanWeights[i] = invValid;
        }

        // 4. Upload and dispatch encoder stack + GPU mean-pool.
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

            // Optional GPU L2-Normalize on the H-element pooled vector,
            // composed from GEMM (sum-of-squares) + SQRT (ε² folded into
            // ScaleBias) + DIVIDE (broadcast). The CPU now only sees the
            // already-normalised H floats and never touches the result.
            GpuBuffer downloadFrom;
            if (request.normalize()) {
                DirectMlTensor normT = tensorOf(normalizedBuf, H);
                l2Kernel.dispatch(pooledT, normT, 1e-12f);
                downloadFrom = normalizedBuf;
            } else {
                downloadFrom = entry.pooled;
            }

            // Read back only the (optionally normalised) H-vector – the
            // [B,H] hidden tensor never crosses the PCIe boundary anymore.
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

        int H = config.hiddenSize();
        long hiddenBytes = (long) bucket * H * Float.BYTES;
        long maskBytes = (long) bucket * Float.BYTES;
        long pooledBytes = (long) H * Float.BYTES;

        GpuBuffer xIn = null, xOut = null, mask = null, meanWeights = null, pooled = null;
        DirectMlMiniLmEncoderStack stack = null;
        DirectMlMeanPoolKernel meanPool = null;
        try {
            xIn  = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            xOut = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            mask = ctx.allocateBuffer(maskBytes,   GpuBuffer.BufferUsage.ACTIVATION);
            meanWeights = ctx.allocateBuffer(maskBytes, GpuBuffer.BufferUsage.ACTIVATION);
            pooled = ctx.allocateBuffer(pooledBytes, GpuBuffer.BufferUsage.ACTIVATION);
            stack = new DirectMlMiniLmEncoderStack(
                    ctx, bucket, H,
                    config.numAttentionHeads(), config.headDim(),
                    config.intermediateSize(),  config.numLayers(),
                    config.layerNormEps(),      /* hasMask */ true);
            meanPool = new DirectMlMeanPoolKernel(ctx, bucket, H);
            StackEntry entry = new StackEntry(stack, xIn, xOut, mask, meanWeights, pooled, meanPool);
            stackCache.put(bucket, entry);
            log.info("DirectMlMiniLmEncoder stack ready for bucket S={} (cached buckets so far: {})",
                    bucket, stackCache.size());
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

    /**
     * Number of cached form-bound stacks (one per active pad bucket).
     */
    public int cachedStackCount() { return stackCache.size(); }
}

