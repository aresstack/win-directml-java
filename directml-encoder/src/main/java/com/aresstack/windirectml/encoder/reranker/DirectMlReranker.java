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
 * keeps the classification head on the CPU - a {@code [1,H]} GEMM that
 * is essentially free compared to a full BERT pass, and that lets the
 * GPU graph stay identical to the embedding pipeline.
 * <p>
 * Pipeline:
 * <pre>
 *   request(docs) -> group documents by pad-bucket B = bucketFor(len)
 *               -> per bucket build batched embeddings [N*B, H] + mask [N*B]
 *               -> upload
 *               -> DirectMlBertEncoderStack (embLN + N layers) with batch=N
 *               -> readback output [N*B, H], slice N CLS rows on CPU
 *                  (rows live at offsets 0, B*H, 2*B*H, ...)
 *               -> CPU classifier (W*cls + b) per row
 *               -> score
 * </pre>
 * Pad buckets are shared with {@link DirectMlBertEncoder}. One stack is
 * cached per active {@code (bucket, batch)} pair, so a steady reranker
 * workload with comparable top-N values reuses the same DirectML graph.
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
    /**
     * Stack cache keyed by {@code (bucket << 32) | batch}. One DirectML
     * graph is reused per active {@code (S, N)} pair - a typical rerank
     * request only spans a handful of pairs (1-3 distinct buckets, N
     * usually = topN), so the cache stays small but reuse is high.
     */
    private final Map<Long, StackEntry> stackCache = new HashMap<>();
    /**
     * Reusable float[H] scratch the CPU classifier reads CLS rows into.
     */
    private final float[] clsScratch;
    private volatile boolean ready;

    private record StackEntry(int bucket, int batch,
                              DirectMlBertEncoderStack stack,
                              GpuBuffer xIn, GpuBuffer xOut, GpuBuffer mask,
                              CpuTensor readback) implements AutoCloseable {
        @Override
        public void close() {
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
            embLnBetaBuf = uploadVec(bert.embLnBeta, H);
            List<BertGpuLayerWeights> built = new ArrayList<>(bert.layers.size());
            for (BertCpuLayerWeights l : bert.layers) {
                built.add(new BertGpuLayerWeights(
                        uploadMat(l.qWeight(), H, H), uploadVec(l.qBias(), H),
                        uploadMat(l.kWeight(), H, H), uploadVec(l.kBias(), H),
                        uploadMat(l.vWeight(), H, H), uploadVec(l.vBias(), H),
                        uploadMat(l.attnOutWeight(), H, H), uploadVec(l.attnOutBias(), H),
                        uploadVec(l.attnLnGamma(), H), uploadVec(l.attnLnBeta(), H),
                        uploadMat(l.mlpInterWeight(), I, H), uploadVec(l.mlpInterBias(), I),
                        uploadMat(l.mlpOutWeight(), H, I), uploadVec(l.mlpOutBias(), H),
                        uploadVec(l.outLnGamma(), H), uploadVec(l.outLnBeta(), H)));
            }
            this.gpuLayers = List.copyOf(built);
            this.clsScratch = new float[H];
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

    @Override
    public boolean isReady() {
        return ready;
    }

    @Override
    public String modelName() {
        return cfg.modelName();
    }

    @Override
    public List<RerankResult> rerank(RerankRequest request) throws RerankException {
        Objects.requireNonNull(request, "request");
        if (!ready) throw new RerankException("DirectMlReranker is closed");
        List<String> docs = request.documents();
        int n = docs.size();
        int H = cfg.hiddenSize();
        BertCpuEncoderWeights bert = weights.bert();
        // 1. Tokenise all pairs and bucket them by pad-bucket B.
        EncoderTokenizer.Encoded[] encs = new EncoderTokenizer.Encoded[n];
        Map<Integer, List<Integer>> byBucket = new HashMap<>();
        for (int i = 0; i < n; i++) {
            EncoderTokenizer.Encoded enc;
            try {
                enc = tokenizer.encodePair(request.query(), docs.get(i));
            } catch (UnsupportedOperationException e) {
                throw new RerankException("Tokenizer does not support pair encoding", e);
            }
            if (enc.length() < 3) {
                throw new RerankException("encodePair produced degenerate sequence of length "
                        + enc.length() + " at index " + i);
            }
            encs[i] = enc;
            int b = DirectMlBertEncoder.bucketFor(enc.length());
            byBucket.computeIfAbsent(b, k -> new ArrayList<>()).add(i);
        }
        double[] scores = new double[n];
        // 2. One batched DirectML dispatch per (bucket, group-size).
        for (Map.Entry<Integer, List<Integer>> e : byBucket.entrySet()) {
            int B = e.getKey();
            List<Integer> group = e.getValue();
            int N = group.size();
            // Build batched [N*B, H] embeddings and [N*B] additive mask on
            // the CPU. The encoder stack treats the leading dimension as
            // rows = batch * seq for row-wise ops and as batch = N for
            // attention/head-layout, so the layout
            //   doc0[B,H] | doc1[B,H] | ...
            // lines up exactly with the batched contract documented in
            // DirectMlBertEncoderStack.
            float[] x = new float[N * B * H];
            float[] mask = new float[N * B];
            for (int gi = 0; gi < N; gi++) {
                int docIdx = group.get(gi);
                EncoderTokenizer.Encoded enc = encs[docIdx];
                float[] xi = BertEmbeddingLookup.lookup(cfg, bert.wordEmbeddings,
                        bert.positionEmbeddings, bert.tokenTypeEmbeddings, enc, B);
                System.arraycopy(xi, 0, x, gi * B * H, B * H);
                float[] mi = BertPoolingWeights.additiveMask(enc.attentionMask(), enc.length(), B);
                System.arraycopy(mi, 0, mask, gi * B, B);
            }
            StackEntry entry;
            try {
                entry = stackFor(B, N);
            } catch (DirectMlRuntimeException ex) {
                throw new RerankException("Failed to build DirectML stack for bucket=" + B
                        + ", batch=" + N, ex);
            }
            try {
                CpuTensor xCpu = CpuTensor.float32(TensorShape.of(N * B, H), x);
                CpuTensor maskCpu = CpuTensor.float32(TensorShape.of(N * B), mask);
                entry.xIn.upload(xCpu);
                entry.mask.upload(maskCpu);
                DirectMlTensor xInT = tensorOf(entry.xIn, N * B, H);
                DirectMlTensor xOutT = tensorOf(entry.xOut, N * B, H);
                DirectMlTensor maskT = tensorOf(entry.mask, N * B);
                DirectMlTensor embGT = tensorOf(embLnGammaBuf, H);
                DirectMlTensor embBT = tensorOf(embLnBetaBuf, H);
                entry.stack.dispatch(xInT, embGT, embBT, gpuLayers, maskT, xOutT);
                // The download path supports partial reads (n * 4 bytes
                // from offset 0) but not strided ones, so we read the
                // full [N*B, H] output back and pick out the N CLS rows
                // on the CPU. The bandwidth cost still amortises across
                // N pairs and the GPU has already saved N-1 dispatches.
                entry.xOut.download(entry.readback);
                FloatBuffer fv = entry.readback.data().duplicate()
                        .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                for (int gi = 0; gi < N; gi++) {
                    fv.position(gi * B * H);
                    fv.get(clsScratch, 0, H);
                    scores[group.get(gi)] = applyClassifier(clsScratch);
                }
            } catch (DirectMlRuntimeException ex) {
                throw new RerankException("DirectML dispatch failed for bucket=" + B
                        + ", batch=" + N, ex);
            }
        }
        List<RerankResult> scored = new ArrayList<>(n);
        for (int i = 0; i < n; i++) scored.add(new RerankResult(i, scores[i]));
        scored.sort(Comparator.comparingDouble(RerankResult::score).reversed());
        int top = request.effectiveTopN();
        return top >= scored.size() ? scored : scored.subList(0, top);
    }

    /**
     * Number of cached form-bound stacks - one entry per active
     * {@code (bucket, batch)} pair seen so far.
     */
    public int cachedStackCount() {
        return stackCache.size();
    }

    private double applyClassifier(float[] clsHidden) {
        int H = cfg.hiddenSize();
        double sum = weights.classifierBias[0];
        float[] w = weights.classifierWeight;
        for (int i = 0; i < H; i++) sum += (double) w[i] * clsHidden[i];
        return sum;
    }

    private synchronized StackEntry stackFor(int bucket, int batch) throws DirectMlRuntimeException {
        long key = ((long) bucket << 32) | (batch & 0xffffffffL);
        StackEntry cached = stackCache.get(key);
        if (cached != null) return cached;
        int H = cfg.hiddenSize();
        long rowsBytes = (long) batch * bucket * H * Float.BYTES;
        long maskBytes = (long) batch * bucket * Float.BYTES;
        GpuBuffer xIn = null, xOut = null, mask = null;
        DirectMlBertEncoderStack stack = null;
        try {
            xIn = ctx.allocateBuffer(rowsBytes, GpuBuffer.BufferUsage.ACTIVATION);
            xOut = ctx.allocateBuffer(rowsBytes, GpuBuffer.BufferUsage.ACTIVATION);
            mask = ctx.allocateBuffer(maskBytes, GpuBuffer.BufferUsage.ACTIVATION);
            stack = new DirectMlBertEncoderStack(ctx, batch, bucket, H,
                    cfg.numHeads(), cfg.headDim(), cfg.intermediateSize(),
                    cfg.numLayers(), cfg.layerNormEps(), /* hasMask */ true);
            CpuTensor readback = emptyCpuTensor(batch * bucket * H);
            StackEntry entry = new StackEntry(bucket, batch, stack, xIn, xOut, mask, readback);
            stackCache.put(key, entry);
            log.info("DirectMlReranker({}) stack ready for bucket S={}, batch N={} (cached so far: {})",
                    cfg.modelName(), bucket, batch, stackCache.size());
            return entry;
        } catch (DirectMlRuntimeException | RuntimeException e) {
            if (stack != null) try {
                stack.close();
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
                    : new DirectMlRuntimeException("Failed to allocate stack for bucket="
                    + bucket + ", batch=" + batch, e);
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
}
