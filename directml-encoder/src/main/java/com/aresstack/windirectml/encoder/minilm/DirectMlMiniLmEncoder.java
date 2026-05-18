package com.aresstack.windirectml.encoder.minilm;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.encoder.EncoderTokenizer;
import com.aresstack.windirectml.encoder.pooling.L2Normalize;
import com.aresstack.windirectml.encoder.pooling.MeanPooling;
import com.aresstack.windirectml.encoder.tokenizer.WordPieceTokenizer;
import com.aresstack.windirectml.runtime.CpuTensor;
import com.aresstack.windirectml.runtime.DirectMlContextImpl;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;
import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.GpuBuffer;
import com.aresstack.windirectml.runtime.TensorDataType;
import com.aresstack.windirectml.runtime.TensorLayout;
import com.aresstack.windirectml.runtime.TensorShape;
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
 *     → Download
 *     → MeanPool (CPU)
 *     → L2-Normalize (CPU, optional je Request)
 *     → EmbeddingVector
 * </pre>
 * Die DirectML-Stacks sind formgebunden auf {@code seq}; eine Map cacht
 * eine {@link StackEntry}-Instanz pro tatsächlich vorgekommener Sequenz-
 * Länge. Ein Pad-Bucket-Cache ({@code S = 64/128/256/512}) ist ein
 * späterer Optimierungs-Sprint.
 * <p>
 * Alle 16 Gewichts-Buffer pro Layer sowie das Embedding-LN-Paar werden
 * im Konstruktor einmal hochgeladen und für die Lebensdauer des Encoders
 * gehalten.
 * <p>
 * <b>Feature levels:</b> {@code DirectMlMiniLmLayerBlock} benötigt
 * {@code DML_FEATURE_LEVEL_5_1} wegen der fused GELU. Auf Windows-11-
 * In-Box (FL 5.0) schlägt die Konstruktion mit {@code E_INVALIDARG}
 * fehl – per {@code -Dwindirectml.directml.dll=...} kann eine
 * redistributable DLL eingehängt werden. Ein Composite-GELU-Fallback
 * für FL 2.0/5.0 ist ein nachgelagerter Sprint.
 */
public final class DirectMlMiniLmEncoder implements EmbeddingModel, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlMiniLmEncoder.class);

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

    // Per-sequence-length stack cache. Stacks are form-bound, so each
    // unique seq triggers one entry; weights are shared across entries.
    private final Map<Integer, StackEntry> stackCache = new HashMap<>();

    private volatile boolean ready = false;

    private record StackEntry(DirectMlMiniLmEncoderStack stack,
                              GpuBuffer xIn,
                              GpuBuffer xOut,
                              GpuBuffer mask) implements AutoCloseable {
        @Override public void close() {
            try { stack.close(); } catch (Exception ignored) {}
            try { xIn.close();   } catch (Exception ignored) {}
            try { xOut.close();  } catch (Exception ignored) {}
            try { mask.close();  } catch (Exception ignored) {}
        }
    }

    /**
     * Construct an encoder that owns its own {@link DirectMlContextImpl}.
     * The context is initialised eagerly; FL 5.1 (fused GELU) is required.
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
            if (!DirectMlBindings.supportsFusedGelu(fl)) {
                throw new EmbeddingException("DirectMlMiniLmEncoder requires "
                        + "DML_FEATURE_LEVEL_5_1 (fused GELU), got "
                        + DirectMlBindings.formatFeatureLevel(fl)
                        + " — pass -Dwindirectml.directml.dll=<redistributable> "
                        + "or wait for a composite GELU fallback sprint");
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
        // 1. CPU embedding lookup (word + pos + tokenType).
        float[] x = new float[seqLen * H];
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
        // 2. Build/get cached stack and per-seq scratch buffers.
        StackEntry entry;
        try {
            entry = stackFor(seqLen);
        } catch (DirectMlRuntimeException e) {
            throw new EmbeddingException("Failed to build DirectML encoder stack for seq=" + seqLen, e);
        }

        // 3. Mask: 0.0 valid, -1e9 padding.
        float[] mask = new float[seqLen];
        for (int i = 0; i < seqLen; i++) {
            mask[i] = encoded.attentionMask()[i] == 0 ? -1e9f : 0f;
        }

        // 4. Upload and dispatch.
        float[] xOutGpu = new float[seqLen * H];
        try {
            CpuTensor xCpu    = CpuTensor.float32(TensorShape.of(seqLen, H), x);
            CpuTensor maskCpu = CpuTensor.float32(TensorShape.of(seqLen), mask);
            entry.xIn.upload(xCpu);
            entry.mask.upload(maskCpu);

            DirectMlTensor xInT   = tensorOf(entry.xIn,  seqLen, H);
            DirectMlTensor xOutT  = tensorOf(entry.xOut, seqLen, H);
            DirectMlTensor maskT  = tensorOf(entry.mask, seqLen);
            DirectMlTensor embGT  = tensorOf(embLnGammaBuf, H);
            DirectMlTensor embBT  = tensorOf(embLnBetaBuf,  H);

            entry.stack.dispatch(xInT, embGT, embBT, gpuLayers, maskT, xOutT);

            CpuTensor xOutCpu = emptyCpuTensor(seqLen * H);
            entry.xOut.download(xOutCpu);
            FloatBuffer fv = xOutCpu.data().duplicate().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            fv.position(0);
            fv.get(xOutGpu, 0, seqLen * H);
        } catch (DirectMlRuntimeException e) {
            throw new EmbeddingException("DirectML dispatch failed for seq=" + seqLen, e);
        }

        // 5. Mean-Pool over attention mask.
        float[][] tokens = new float[seqLen][H];
        for (int t = 0; t < seqLen; t++) {
            System.arraycopy(xOutGpu, t * H, tokens[t], 0, H);
        }
        float[] pooled = MeanPooling.pool(tokens, encoded.attentionMask());

        // 6. L2-Normalize.
        if (request.normalize()) {
            L2Normalize.inPlace(pooled, 1e-12f);
        }
        return new EmbeddingVector(pooled, H, MiniLmArchitecture.NAME, request.normalize());
    }

    // ── helpers ─────────────────────────────────────────────────────────

    private synchronized StackEntry stackFor(int seqLen) throws DirectMlRuntimeException {
        StackEntry cached = stackCache.get(seqLen);
        if (cached != null) return cached;

        int H = config.hiddenSize();
        long hiddenBytes = (long) seqLen * H * Float.BYTES;
        long maskBytes   = (long) seqLen * Float.BYTES;

        GpuBuffer xIn = null, xOut = null, mask = null;
        DirectMlMiniLmEncoderStack stack = null;
        try {
            xIn  = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            xOut = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            mask = ctx.allocateBuffer(maskBytes,   GpuBuffer.BufferUsage.ACTIVATION);
            stack = new DirectMlMiniLmEncoderStack(
                    ctx, seqLen, H,
                    config.numAttentionHeads(), config.headDim(),
                    config.intermediateSize(),  config.numLayers(),
                    config.layerNormEps(),      /* hasMask */ true);
            StackEntry entry = new StackEntry(stack, xIn, xOut, mask);
            stackCache.put(seqLen, entry);
            return entry;
        } catch (DirectMlRuntimeException | RuntimeException e) {
            if (stack != null) try { stack.close(); } catch (Exception ignored) {}
            if (mask != null)  try { mask.close();  } catch (Exception ignored) {}
            if (xOut != null)  try { xOut.close();  } catch (Exception ignored) {}
            if (xIn != null)   try { xIn.close();   } catch (Exception ignored) {}
            throw (e instanceof DirectMlRuntimeException d) ? d
                    : new DirectMlRuntimeException("Failed to allocate stack for seq=" + seqLen, e);
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

    /** Number of cached form-bound stacks (one per encountered seq length). */
    public int cachedStackCount() { return stackCache.size(); }
}

