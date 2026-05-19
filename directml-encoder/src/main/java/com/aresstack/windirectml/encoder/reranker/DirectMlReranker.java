package com.aresstack.windirectml.encoder.reranker;

import com.aresstack.windirectml.encoder.EncoderTokenizer;
import com.aresstack.windirectml.encoder.bert.BertCpuEncoderWeights;
import com.aresstack.windirectml.encoder.bert.BertCpuLayerWeights;
import com.aresstack.windirectml.encoder.bert.BertEmbeddingLookup;
import com.aresstack.windirectml.encoder.bert.BertEncoderConfig;
import com.aresstack.windirectml.encoder.bert.BertGpuLayerWeights;
import com.aresstack.windirectml.encoder.bert.BertPoolingWeights;
import com.aresstack.windirectml.encoder.bert.DirectMlBertEncoder;
import com.aresstack.windirectml.encoder.bert.DirectMlBertEncoderStack;
import com.aresstack.windirectml.runtime.CpuTensor;
import com.aresstack.windirectml.runtime.DirectMlContextImpl;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;
import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.GpuBuffer;
import com.aresstack.windirectml.runtime.TensorDataType;
import com.aresstack.windirectml.runtime.TensorLayout;
import com.aresstack.windirectml.runtime.TensorShape;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * DirectML cross-encoder reranker. Reuses the generic
 * {@link DirectMlBertEncoderStack} (embedding-LN + N layer blocks) and
 * keeps the classification head on the CPU – a {@code [1,H]} GEMM that
 * is essentially free compared to a full BERT pass, and that lets the
 * GPU graph stay identical to the embedding pipeline.
 * <p>
 * Pipeline:
 * <pre>
 *   (query, doc) → tokenizer.encodePair → CPU embedding lookup
 *               → upload [B,H]
 *               → DirectMlBertEncoderStack (embLN + N layers)
 *               → download [B,H]
 *               → slice CLS row [H]
 *               → CPU classifier (W·cls + b)
 *               → score
 * </pre>
 * Pad buckets are shared with {@link DirectMlBertEncoder} so a
 * mixed-length batch of candidate documents amortises stack-cache
 * allocation across calls.
 */
public final class DirectMlReranker implements Reranker {

    private static final Logger log = LoggerFactory.getLogger(DirectMlReranker.class);

    private final BertEncoderConfig cfg;
    private final RerankerCpuWeights weights;
    private final EncoderTokenizer tokenizer;
    private final DirectMlContextImpl ctx;
    private final boolean ownsCtx;

    private final List<GpuBuffer> ownedGpuBuffers = new ArrayList<>();
    private final GpuBuffer embLnGammaBuf;
    private final GpuBuffer embLnBetaBuf;
    private final List<BertGpuLayerWeights> gpuLayers;

    private final Map<Integer, StackEntry> stackCache = new HashMap<>();
    private volatile boolean ready;

    private record StackEntry(DirectMlBertEncoderStack stack,
                              GpuBuffer xIn, GpuBuffer xOut, GpuBuffer mask) implements AutoCloseable {
        @Override public void close() {
            try { stack.close(); } catch (Exception ignored) {}
            try { xIn.close();   } catch (Exception ignored) {}
            try { xOut.close();  } catch (Exception ignored) {}
            try { mask.close();  } catch (Exception ignored) {}
        }
    }

    public static DirectMlReranker build(DirectMlContextImpl ctx,
                                         boolean ownsCtx,
                                         RerankerCpuWeights weights,
                                         EncoderTokenizer tokenizer) throws RerankException {
        return new DirectMlReranker(ctx, ownsCtx, weights, tokenizer);
    }

    private DirectMlReranker(DirectMlContextImpl ctx, boolean ownsCtx,
                             RerankerCpuWeights weights, EncoderTokenizer tokenizer)
            throws RerankException {
        this.ctx = Objects.requireNonNull(ctx);
        this.ownsCtx = ownsCtx;
        this.weights = Objects.requireNonNull(weights);
        this.tokenizer = Objects.requireNonNull(tokenizer);
        this.cfg = weights.config();
        this.cfg.validate();

        try {
            if (!WindowsBindings.isSupported()) {
                throw new RerankException("DirectML requires Windows + D3D12 on this host");
            }
            if (!ctx.isReady() || !ctx.bindings().hasDirectMl()) {
                throw new RerankException("DirectML context not ready");
            }
            int H = cfg.hiddenSize();
            int I = cfg.intermediateSize();
            BertCpuEncoderWeights bert = weights.bert();

            embLnGammaBuf = uploadVec(bert.embLnGamma, H);
            embLnBetaBuf  = uploadVec(bert.embLnBeta,  H);

            List<BertGpuLayerWeights> built = new ArrayList<>(bert.layers.size());
            for (BertCpuLayerWeights l : bert.layers) {
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
            this.ready = true;
            log.info("DirectMlReranker ready: model={}, layers={}, hidden={}, heads={}, inter={}",
                    cfg.modelName(), cfg.numLayers(), cfg.hiddenSize(),
                    cfg.numHeads(), cfg.intermediateSize());
        } catch (DirectMlRuntimeException e) {
            closeOwnedQuietly();
            throw new RerankException("Failed to upload reranker weights to GPU for "
                    + cfg.modelName(), e);
        } catch (RuntimeException e) {
            closeOwnedQuietly();
            throw e;
        }
    }

    @Override public boolean isReady() { return ready; }
    @Override public String modelName() { return cfg.modelName(); }

    @Override
    public List<RerankResult> rerank(RerankRequest request) throws RerankException {
        Objects.requireNonNull(request, "request");
        if (!ready) throw new RerankException("DirectMlReranker is closed");
        List<String> docs = request.documents();
        List<RerankResult> scored = new ArrayList<>(docs.size());
        for (int i = 0; i < docs.size(); i++) {
            scored.add(new RerankResult(i, scorePair(request.query(), docs.get(i))));
        }
        scored.sort(Comparator.comparingDouble(RerankResult::score).reversed());
        int top = request.effectiveTopN();
        return top >= scored.size() ? scored : scored.subList(0, top);
    }

    /** Number of cached form-bound stacks (one per active pad bucket). */
    public int cachedStackCount() { return stackCache.size(); }

    private double scorePair(String query, String document) throws RerankException {
        EncoderTokenizer.Encoded enc;
        try {
            enc = tokenizer.encodePair(query, document);
        } catch (UnsupportedOperationException e) {
            throw new RerankException("Tokenizer does not support pair encoding", e);
        }
        int seqLen = enc.length();
        if (seqLen < 3) {
            throw new RerankException("encodePair produced degenerate sequence of length " + seqLen);
        }
        int H = cfg.hiddenSize();
        int B = DirectMlBertEncoder.bucketFor(seqLen);
        BertCpuEncoderWeights bert = weights.bert();
        float[] x = BertEmbeddingLookup.lookup(cfg, bert.wordEmbeddings,
                bert.positionEmbeddings, bert.tokenTypeEmbeddings, enc, B);

        StackEntry entry;
        try {
            entry = stackFor(B);
        } catch (DirectMlRuntimeException e) {
            throw new RerankException("Failed to build DirectML stack for bucket=" + B
                    + " (seqLen=" + seqLen + ")", e);
        }

        float[] mask = BertPoolingWeights.additiveMask(enc.attentionMask(), seqLen, B);
        float[] cls = new float[H];
        try {
            CpuTensor xCpu = CpuTensor.float32(TensorShape.of(B, H), x);
            CpuTensor maskCpu = CpuTensor.float32(TensorShape.of(B), mask);
            entry.xIn.upload(xCpu);
            entry.mask.upload(maskCpu);

            DirectMlTensor xInT = tensorOf(entry.xIn, B, H);
            DirectMlTensor xOutT = tensorOf(entry.xOut, B, H);
            DirectMlTensor maskT = tensorOf(entry.mask, B);
            DirectMlTensor embGT = tensorOf(embLnGammaBuf, H);
            DirectMlTensor embBT = tensorOf(embLnBetaBuf, H);
            entry.stack.dispatch(xInT, embGT, embBT, gpuLayers, maskT, xOutT);

            // Download just the [CLS] row. The current GpuBuffer API does
            // not expose partial download, so we materialise the full [B,H]
            // and slice on the CPU; even at B=512, H=768 this is ~1.5 MB
            // per pair – negligible compared to the encoder pass.
            CpuTensor outCpu = emptyCpuTensor(B * H);
            entry.xOut.download(outCpu);
            FloatBuffer fv = outCpu.data().duplicate().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            fv.position(0);
            fv.get(cls, 0, H);
        } catch (DirectMlRuntimeException e) {
            throw new RerankException("DirectML dispatch failed for bucket=" + B
                    + " (seqLen=" + seqLen + ")", e);
        }

        return applyClassifier(cls);
    }

    private double applyClassifier(float[] clsHidden) {
        int H = cfg.hiddenSize();
        double sum = weights.classifierBias[0];
        float[] w = weights.classifierWeight;
        for (int i = 0; i < H; i++) sum += (double) w[i] * clsHidden[i];
        return sum;
    }

    // ── allocation helpers (mirror DirectMlBertEncoder) ───────────────────

    private synchronized StackEntry stackFor(int bucket) throws DirectMlRuntimeException {
        StackEntry cached = stackCache.get(bucket);
        if (cached != null) return cached;
        int H = cfg.hiddenSize();
        long hiddenBytes = (long) bucket * H * Float.BYTES;
        long maskBytes = (long) bucket * Float.BYTES;
        GpuBuffer xIn = null, xOut = null, mask = null;
        DirectMlBertEncoderStack stack = null;
        try {
            xIn  = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            xOut = ctx.allocateBuffer(hiddenBytes, GpuBuffer.BufferUsage.ACTIVATION);
            mask = ctx.allocateBuffer(maskBytes,   GpuBuffer.BufferUsage.ACTIVATION);
            stack = new DirectMlBertEncoderStack(ctx, bucket, H,
                    cfg.numHeads(), cfg.headDim(), cfg.intermediateSize(),
                    cfg.numLayers(), cfg.layerNormEps(), /* hasMask */ true);
            StackEntry entry = new StackEntry(stack, xIn, xOut, mask);
            stackCache.put(bucket, entry);
            log.info("DirectMlReranker({}) stack ready for bucket S={} (cached so far: {})",
                    cfg.modelName(), bucket, stackCache.size());
            return entry;
        } catch (DirectMlRuntimeException | RuntimeException e) {
            if (stack != null) try { stack.close(); } catch (Exception ignored) {}
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
}

