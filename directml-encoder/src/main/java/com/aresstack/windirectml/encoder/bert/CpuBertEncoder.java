package com.aresstack.windirectml.encoder.bert;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.encoder.EncoderTokenizer;
import com.aresstack.windirectml.encoder.pooling.L2Normalize;
import com.aresstack.windirectml.encoder.pooling.MeanPooling;

import java.util.Objects;

/**
 * Generic CPU implementation of a BERT-style sentence-embedding encoder
 * (MiniLM, E5-v2, BGE, …). Math is identical to the original
 * {@code CpuMiniLmEncoder} (Embedding-LN → N layers of
 * Q/K/V → MHA → Wo+LN → MLP+GELU+LN → mean pool → L2); only the
 * config and weight types are model-agnostic.
 * <p>
 * Used both as the CPU backend ({@code -Dembed.backend=cpu}) and as
 * the parity reference for {@link DirectMlBertEncoder}.
 */
public final class CpuBertEncoder implements EmbeddingModel, AutoCloseable {

    private final BertEncoderConfig cfg;
    private final BertCpuEncoderWeights weights;
    private final EncoderTokenizer tokenizer;
    private volatile boolean ready;

    public CpuBertEncoder(BertEncoderConfig cfg,
                          BertCpuEncoderWeights weights,
                          EncoderTokenizer tokenizer) {
        this.cfg = Objects.requireNonNull(cfg);
        this.weights = Objects.requireNonNull(weights);
        this.tokenizer = Objects.requireNonNull(tokenizer);
        this.cfg.validate();
        this.ready = true;
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
        String text = request.prefix() != null ? request.prefix() + request.text() : request.text();
        EncoderTokenizer.Encoded encoded = tokenizer.encode(text);
        int seqLen = encoded.length();
        if (seqLen < 2) {
            throw new EmbeddingException("Tokenization produced empty sequence");
        }

        int H = cfg.hiddenSize();
        int I = cfg.intermediateSize();
        int nh = cfg.numHeads();
        int hd = cfg.headDim();
        float eps = cfg.layerNormEps();

        // 1. Embedding lookup (word + position + tokenType)
        float[] x = new float[seqLen * H];
        for (int t = 0; t < seqLen; t++) {
            int id = encoded.inputIds()[t];
            int tt = encoded.tokenTypeIds()[t];
            int dst = t * H, ws = id * H, ps = t * H, tts = tt * H;
            for (int h = 0; h < H; h++) {
                x[dst + h] = weights.wordEmbeddings[ws + h]
                        + weights.positionEmbeddings[ps + h]
                        + weights.tokenTypeEmbeddings[tts + h];
            }
        }
        layerNormInPlace(x, seqLen, H, weights.embLnGamma, weights.embLnBeta, eps);

        // 2. Transformer layers
        float[] q = new float[seqLen * H];
        float[] k = new float[seqLen * H];
        float[] v = new float[seqLen * H];
        float[] attn = new float[seqLen * H];
        float[] attnOut = new float[seqLen * H];
        float[] mlpInter = new float[seqLen * I];
        float[] mlpOut = new float[seqLen * H];
        float[] scores = new float[seqLen * seqLen];

        for (BertCpuLayerWeights lw : weights.layers) {
            forwardSingleLayer(x, lw, encoded.attentionMask(), seqLen,
                    H, nh, hd, I, eps,
                    q, k, v, attn, attnOut, mlpInter, mlpOut, scores);
        }

        // 3. Mean pooling over attention mask
        float[][] tokens = new float[seqLen][H];
        for (int t = 0; t < seqLen; t++) {
            System.arraycopy(x, t * H, tokens[t], 0, H);
        }
        float[] pooled = MeanPooling.pool(tokens, encoded.attentionMask());

        // 4. L2-Normalize (optional)
        if (request.normalize()) {
            L2Normalize.inPlace(pooled, 1e-12f);
        }

        return new EmbeddingVector(pooled, H, cfg.modelName(), request.normalize());
    }

    // ── shared BERT math (also reusable by tests) ───────────────────────

    public static void forwardSingleLayer(float[] x,
                                          BertCpuLayerWeights lw,
                                          int[] attentionMask,
                                          int seqLen,
                                          int hiddenSize, int numHeads, int headDim,
                                          int intermediateSize, float layerNormEps,
                                          float[] q, float[] k, float[] v,
                                          float[] attn, float[] attnOut,
                                          float[] mlpInter, float[] mlpOut,
                                          float[] scoresBuf) {
        matmulXWtPlusB(x, seqLen, hiddenSize, lw.qWeight(), hiddenSize, lw.qBias(), q);
        matmulXWtPlusB(x, seqLen, hiddenSize, lw.kWeight(), hiddenSize, lw.kBias(), k);
        matmulXWtPlusB(x, seqLen, hiddenSize, lw.vWeight(), hiddenSize, lw.vBias(), v);

        multiHeadAttention(q, k, v, attentionMask, seqLen,
                hiddenSize, numHeads, headDim, attn, scoresBuf);

        matmulXWtPlusB(attn, seqLen, hiddenSize, lw.attnOutWeight(), hiddenSize, lw.attnOutBias(), attnOut);

        for (int i = 0; i < seqLen * hiddenSize; i++) x[i] = x[i] + attnOut[i];
        layerNormInPlace(x, seqLen, hiddenSize, lw.attnLnGamma(), lw.attnLnBeta(), layerNormEps);

        matmulXWtPlusB(x, seqLen, hiddenSize, lw.mlpInterWeight(), intermediateSize, lw.mlpInterBias(), mlpInter);
        geluInPlace(mlpInter);
        matmulXWtPlusB(mlpInter, seqLen, intermediateSize, lw.mlpOutWeight(), hiddenSize, lw.mlpOutBias(), mlpOut);

        for (int i = 0; i < seqLen * hiddenSize; i++) x[i] = x[i] + mlpOut[i];
        layerNormInPlace(x, seqLen, hiddenSize, lw.outLnGamma(), lw.outLnBeta(), layerNormEps);
    }

    private static void matmulXWtPlusB(float[] x, int M, int K,
                                       float[] w, int N,
                                       float[] bias, float[] out) {
        for (int m = 0; m < M; m++) {
            int xi = m * K, oi = m * N;
            for (int n = 0; n < N; n++) {
                int wi = n * K;
                float sum = 0f;
                for (int kk = 0; kk < K; kk++) sum += x[xi + kk] * w[wi + kk];
                out[oi + n] = bias != null ? sum + bias[n] : sum;
            }
        }
    }

    private static void layerNormInPlace(float[] x, int M, int H,
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

    private static void geluInPlace(float[] x) {
        for (int i = 0; i < x.length; i++) {
            float xi = x[i];
            x[i] = (float) (0.5 * xi * (1.0 + erf(xi / Math.sqrt(2.0))));
        }
    }

    private static double erf(double xRaw) {
        double sign = xRaw < 0 ? -1.0 : 1.0;
        double x = Math.abs(xRaw);
        double t = 1.0 / (1.0 + 0.3275911 * x);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
        return sign * y;
    }

    private static void multiHeadAttention(float[] q, float[] k, float[] v,
                                           int[] attentionMask, int seqLen,
                                           int hiddenSize, int numHeads, int headDim,
                                           float[] out, float[] scoresBuf) {
        double scale = 1.0 / Math.sqrt(headDim);
        for (int h = 0; h < numHeads; h++) {
            int headOffset = h * headDim;
            for (int i = 0; i < seqLen; i++) {
                int qBase = i * hiddenSize + headOffset;
                for (int j = 0; j < seqLen; j++) {
                    int kBase = j * hiddenSize + headOffset;
                    double sum = 0.0;
                    for (int d = 0; d < headDim; d++) sum += q[qBase + d] * k[kBase + d];
                    double score = sum * scale;
                    if (attentionMask[j] == 0) score = -1e9;
                    scoresBuf[i * seqLen + j] = (float) score;
                }
            }
            for (int i = 0; i < seqLen; i++) {
                int row = i * seqLen;
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < seqLen; j++) if (scoresBuf[row + j] > max) max = scoresBuf[row + j];
                double sum = 0.0;
                for (int j = 0; j < seqLen; j++) {
                    double e = Math.exp(scoresBuf[row + j] - max);
                    scoresBuf[row + j] = (float) e;
                    sum += e;
                }
                float inv = (float) (1.0 / Math.max(sum, 1e-12));
                for (int j = 0; j < seqLen; j++) scoresBuf[row + j] *= inv;
            }
            for (int i = 0; i < seqLen; i++) {
                int outBase = i * hiddenSize + headOffset;
                for (int d = 0; d < headDim; d++) {
                    double sum = 0.0;
                    for (int j = 0; j < seqLen; j++) {
                        sum += scoresBuf[i * seqLen + j] * v[j * hiddenSize + headOffset + d];
                    }
                    out[outBase + d] = (float) sum;
                }
            }
        }
    }

    @Override
    public void close() {
        ready = false;
    }
}

