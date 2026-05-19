package com.aresstack.windirectml.encoder.reranker;

import com.aresstack.windirectml.encoder.EncoderTokenizer;
import com.aresstack.windirectml.encoder.bert.BertCpuEncoderWeights;
import com.aresstack.windirectml.encoder.bert.BertCpuLayerWeights;
import com.aresstack.windirectml.encoder.bert.BertEncoderConfig;
import com.aresstack.windirectml.encoder.bert.CpuBertEncoder;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * CPU reference reranker. Performs:
 * <ol>
 *   <li>{@link EncoderTokenizer#encodePair} on {@code (query, doc)};</li>
 *   <li>embedding sum + N transformer layers (re-uses
 *       {@link CpuBertEncoder#forwardSingleLayer});</li>
 *   <li>extracts the {@code [CLS]} hidden state (row 0 of the encoder
 *       output) – cross-encoders never mean-pool;</li>
 *   <li>applies the {@code [1, H]} linear classification head;</li>
 *   <li>sorts pairs by score and trims to {@code topN}.</li>
 * </ol>
 * Used both as the {@code -Drerank.backend=cpu} backend and as the
 * parity reference for {@link DirectMlReranker}.
 */
public final class CpuReranker implements Reranker {

    private final BertEncoderConfig cfg;
    private final RerankerCpuWeights weights;
    private final EncoderTokenizer tokenizer;
    private volatile boolean ready;

    public CpuReranker(RerankerCpuWeights weights, EncoderTokenizer tokenizer) {
        this.weights = Objects.requireNonNull(weights);
        this.tokenizer = Objects.requireNonNull(tokenizer);
        this.cfg = weights.config();
        this.cfg.validate();
        this.ready = true;
    }

    @Override public boolean isReady() { return ready; }
    @Override public String modelName() { return cfg.modelName(); }

    @Override
    public List<RerankResult> rerank(RerankRequest request) throws RerankException {
        Objects.requireNonNull(request, "request");
        if (!ready) throw new RerankException("CpuReranker is closed");
        List<String> docs = request.documents();
        List<RerankResult> scored = new ArrayList<>(docs.size());
        for (int i = 0; i < docs.size(); i++) {
            double score = scorePair(request.query(), docs.get(i));
            scored.add(new RerankResult(i, score));
        }
        scored.sort(Comparator.comparingDouble(RerankResult::score).reversed());
        int top = request.effectiveTopN();
        return top >= scored.size() ? scored : scored.subList(0, top);
    }

    /** Single pair forward-pass; returns the raw classifier logit. */
    public double scorePair(String query, String document) throws RerankException {
        EncoderTokenizer.Encoded enc;
        try {
            enc = tokenizer.encodePair(query, document);
        } catch (UnsupportedOperationException e) {
            throw new RerankException("Tokenizer does not support pair encoding", e);
        }
        int seqLen = enc.length();
        if (seqLen < 3) {
            throw new RerankException("encodePair produced a degenerate sequence of length " + seqLen);
        }
        float[] cls = forwardClsHidden(enc);
        return applyClassifier(cls);
    }

    /**
     * Runs the encoder forward pass and returns the {@code [CLS]}
     * hidden state ({@code float[H]}). Visible for parity tests.
     */
    public float[] forwardClsHidden(EncoderTokenizer.Encoded encoded) {
        int seqLen = encoded.length();
        int H = cfg.hiddenSize();
        int I = cfg.intermediateSize();
        int nh = cfg.numHeads();
        int hd = cfg.headDim();
        float eps = cfg.layerNormEps();
        BertCpuEncoderWeights bert = weights.bert();

        // Embedding lookup with token_type_ids respected (segment 0/1).
        float[] x = new float[seqLen * H];
        for (int t = 0; t < seqLen; t++) {
            int id = encoded.inputIds()[t];
            int tt = encoded.tokenTypeIds()[t];
            int dst = t * H, ws = id * H, ps = t * H, tts = tt * H;
            for (int h = 0; h < H; h++) {
                x[dst + h] = bert.wordEmbeddings[ws + h]
                        + bert.positionEmbeddings[ps + h]
                        + bert.tokenTypeEmbeddings[tts + h];
            }
        }
        layerNorm(x, seqLen, H, bert.embLnGamma, bert.embLnBeta, eps);

        float[] q = new float[seqLen * H];
        float[] k = new float[seqLen * H];
        float[] v = new float[seqLen * H];
        float[] attn = new float[seqLen * H];
        float[] attnOut = new float[seqLen * H];
        float[] mlpInter = new float[seqLen * I];
        float[] mlpOut = new float[seqLen * H];
        float[] scores = new float[seqLen * seqLen];

        for (BertCpuLayerWeights lw : bert.layers) {
            CpuBertEncoder.forwardSingleLayer(x, lw, encoded.attentionMask(), seqLen,
                    H, nh, hd, I, eps, q, k, v, attn, attnOut, mlpInter, mlpOut, scores);
        }

        float[] cls = new float[H];
        System.arraycopy(x, 0, cls, 0, H);
        return cls;
    }

    /**
     * Applies the {@code [1,H]} classifier head: {@code score = W · cls + b}.
     */
    public double applyClassifier(float[] clsHidden) {
        int H = cfg.hiddenSize();
        if (clsHidden.length != H) {
            throw new IllegalArgumentException("clsHidden length " + clsHidden.length + " != " + H);
        }
        double sum = weights.classifierBias[0];
        float[] w = weights.classifierWeight;
        for (int i = 0; i < H; i++) sum += (double) w[i] * clsHidden[i];
        return sum;
    }

    // LayerNorm exposed locally so we don't depend on CpuBertEncoder's private
    // helper for the embedding-LN before the first attention block.
    private static void layerNorm(float[] x, int M, int H,
                                  float[] gamma, float[] beta, float eps) {
        for (int m = 0; m < M; m++) {
            int base = m * H;
            double mean = 0.0;
            for (int h = 0; h < H; h++) mean += x[base + h];
            mean /= H;
            double var = 0.0;
            for (int h = 0; h < H; h++) {
                double d = x[base + h] - mean;
                var += d * d;
            }
            var /= H;
            double invStd = 1.0 / Math.sqrt(var + eps);
            for (int h = 0; h < H; h++) {
                double normed = (x[base + h] - mean) * invStd;
                x[base + h] = (float) (normed * gamma[h] + beta[h]);
            }
        }
    }

    @Override
    public void close() { ready = false; }
}

