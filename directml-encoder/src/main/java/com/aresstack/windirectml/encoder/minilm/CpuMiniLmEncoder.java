package com.aresstack.windirectml.encoder.minilm;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.encoder.EncoderTokenizer;
import com.aresstack.windirectml.encoder.pooling.L2Normalize;
import com.aresstack.windirectml.encoder.pooling.MeanPooling;
import com.aresstack.windirectml.encoder.tokenizer.WordPieceTokenizer;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Reine Java/CPU-Implementierung von {@code sentence-transformers/all-MiniLM-L6-v2}.
 * <p>
 * Bewusst korrekt vor performant: alle Matrixmultiplikationen sind dichte
 * triple-nested-Loops. Das reicht für Single-Sentence-Embeddings bei einer
 * Sequenzlänge ≤ 128 in &lt; 100 ms auf einem aktuellen Desktop.
 * <p>
 * Sobald die DirectML-Kernel-Implementierungen verfügbar sind, wird derselbe
 * Forward-Pass über eine {@link com.aresstack.windirectml.runtime.kernels.KernelRegistry}
 * laufen – das CPU-Pfad-Ergebnis dient dann als Referenz.
 * <p>
 * Forward-Schritte (BERT/MiniLM-Standard):
 * <ol>
 *   <li>WordPiece-Tokenisierung mit {@code [CLS] … [SEP]}.</li>
 *   <li>Word- + Position- + TokenType-Embedding-Lookup.</li>
 *   <li>Embedding-LayerNorm.</li>
 *   <li>6 × (Self-Attention → Add+LN → MLP → Add+LN).</li>
 *   <li>Mean-Pooling über die Attention-Mask.</li>
 *   <li>L2-Normalisierung (optional).</li>
 * </ol>
 */
public final class CpuMiniLmEncoder implements EmbeddingModel, AutoCloseable {

    private final MiniLmArchitecture architecture;
    private final MiniLmConfig config;
    private final CpuMiniLmWeights weights;
    private final EncoderTokenizer tokenizer;
    private final float layerNormEps;
    private final int hiddenSize;
    private final int numHeads;
    private final int headDim;
    private final int intermediateSize;
    private final int numLayers;
    private volatile boolean ready;

    public CpuMiniLmEncoder(MiniLmArchitecture architecture,
                            CpuMiniLmWeights weights,
                            EncoderTokenizer tokenizer) {
        this.architecture = Objects.requireNonNull(architecture);
        this.weights = Objects.requireNonNull(weights);
        this.tokenizer = Objects.requireNonNull(tokenizer);
        this.config = architecture.config();
        this.layerNormEps = config.layerNormEps();
        this.hiddenSize = config.hiddenSize();
        this.numHeads = config.numAttentionHeads();
        this.headDim = config.headDim();
        this.intermediateSize = config.intermediateSize();
        this.numLayers = config.numLayers();
        this.ready = true;
    }

    /**
     * Convenience-Loader: liest {@code model.safetensors} und {@code tokenizer.json} aus dem Verzeichnis.
     */
    public static CpuMiniLmEncoder load(Path modelDir) throws EmbeddingException {
        try {
            MiniLmArchitecture arch = new MiniLmArchitecture();
            CpuMiniLmWeights w = CpuMiniLmWeights.load(modelDir, arch);
            WordPieceTokenizer t = WordPieceTokenizer.load(modelDir.resolve("tokenizer.json"),
                    arch.config().maxPositionEmbeddings());
            return new CpuMiniLmEncoder(arch, w, t);
        } catch (EmbeddingException e) {
            throw e;
        } catch (Exception e) {
            throw new EmbeddingException("Failed to load CpuMiniLmEncoder from " + modelDir, e);
        }
    }

    @Override
    public boolean isReady() {
        return ready;
    }

    @Override
    public int dimension() {
        return architecture.outputDimension();
    }

    @Override
    public EmbeddingVector embed(EmbeddingRequest request) throws EmbeddingException {
        String text = request.prefix() != null ? request.prefix() + request.text() : request.text();
        EncoderTokenizer.Encoded encoded = tokenizer.encode(text);
        int seqLen = encoded.length();
        if (seqLen < 2) {
            throw new EmbeddingException("Tokenization produced empty sequence");
        }

        // 1. Embeddings
        float[] x = new float[seqLen * hiddenSize]; // [seq, H]
        for (int t = 0; t < seqLen; t++) {
            int id = encoded.inputIds()[t];
            int tt = encoded.tokenTypeIds()[t];
            int dst = t * hiddenSize;
            int wordSrc = id * hiddenSize;
            int posSrc = t * hiddenSize;
            int ttSrc = tt * hiddenSize;
            for (int h = 0; h < hiddenSize; h++) {
                x[dst + h] = weights.wordEmbeddings[wordSrc + h]
                        + weights.positionEmbeddings[posSrc + h]
                        + weights.tokenTypeEmbeddings[ttSrc + h];
            }
        }
        layerNormInPlace(x, seqLen, hiddenSize, weights.embLnGamma, weights.embLnBeta, layerNormEps);

        // 2. Transformer-Layer
        float[] q = new float[seqLen * hiddenSize];
        float[] k = new float[seqLen * hiddenSize];
        float[] v = new float[seqLen * hiddenSize];
        float[] attn = new float[seqLen * hiddenSize];
        float[] attnOut = new float[seqLen * hiddenSize];
        float[] mlpInter = new float[seqLen * intermediateSize];
        float[] mlpOut = new float[seqLen * hiddenSize];
        float[] scores = new float[seqLen * seqLen]; // per head

        for (int l = 0; l < numLayers; l++) {
            CpuMiniLmWeights.LayerWeights lw = weights.layers.get(l);
            forwardSingleLayer(x, lw, encoded.attentionMask(), seqLen,
                    hiddenSize, numHeads, headDim, intermediateSize, layerNormEps,
                    q, k, v, attn, attnOut, mlpInter, mlpOut, scores);
        }

        // 3. Mean Pooling über Attention-Mask
        float[][] tokens = new float[seqLen][hiddenSize];
        for (int t = 0; t < seqLen; t++) {
            System.arraycopy(x, t * hiddenSize, tokens[t], 0, hiddenSize);
        }
        float[] pooled = MeanPooling.pool(tokens, encoded.attentionMask());

        // 4. L2-Normalize
        if (request.normalize()) {
            L2Normalize.inPlace(pooled, 1e-12f);
        }

        return new EmbeddingVector(pooled, hiddenSize, MiniLmArchitecture.NAME, request.normalize());
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Compute-Hilfsmethoden (CPU)
    // ═════════════════════════════════════════════════════════════════════

    /**
     * Vollständiger Forward-Pass für genau einen MiniLM-Encoder-Block,
     * in-place auf {@code x}.
     * <p>
     * Diese statische Variante ist von {@code DirectMlMiniLmLayerBlockTest}
     * als CPU-Referenz wiederverwendbar – Single-Layer-Vergleich ist der
     * verbindliche Prüfschritt, bevor der vollständige 6-Layer-DirectML-
     * Encoder gebaut wird (siehe {@code docs/head-layout-convention.md}).
     * <p>
     * Reihenfolge der Sub-Schritte exakt nach BERT-Konvention:
     * Q/K/V-Linear → MultiHeadAttention → Output-Projection →
     * Residual-Add → Attn-LayerNorm → MLP-Intermediate-Linear → GELU →
     * MLP-Output-Linear → Residual-Add → Out-LayerNorm.
     * <p>
     * Alle Scratch-Buffer ({@code q, k, v, attn, attnOut, mlpInter,
     * mlpOut, scores}) müssen vom Caller in den passenden Größen
     * vorallokiert übergeben werden, damit diese Methode allokationsfrei
     * läuft.
     */
    public static void forwardSingleLayer(float[] x,
                                          CpuMiniLmWeights.LayerWeights lw,
                                          int[] attentionMask,
                                          int seqLen,
                                          int hiddenSize, int numHeads, int headDim,
                                          int intermediateSize, float layerNormEps,
                                          float[] q, float[] k, float[] v,
                                          float[] attn, float[] attnOut,
                                          float[] mlpInter, float[] mlpOut,
                                          float[] scoresBuf) {
        // Q, K, V projections
        matmulXWtPlusB(x, seqLen, hiddenSize, lw.qWeight, hiddenSize, lw.qBias, q);
        matmulXWtPlusB(x, seqLen, hiddenSize, lw.kWeight, hiddenSize, lw.kBias, k);
        matmulXWtPlusB(x, seqLen, hiddenSize, lw.vWeight, hiddenSize, lw.vBias, v);

        // Multi-Head Attention
        multiHeadAttention(q, k, v, attentionMask, seqLen,
                hiddenSize, numHeads, headDim, attn, scoresBuf);

        // Output projection
        matmulXWtPlusB(attn, seqLen, hiddenSize, lw.attnOutWeight, hiddenSize, lw.attnOutBias, attnOut);

        // Residual + LayerNorm
        for (int i = 0; i < seqLen * hiddenSize; i++) x[i] = x[i] + attnOut[i];
        layerNormInPlace(x, seqLen, hiddenSize, lw.attnLnGamma, lw.attnLnBeta, layerNormEps);

        // MLP: x -> intermediate, GELU, -> hidden
        matmulXWtPlusB(x, seqLen, hiddenSize, lw.mlpInterWeight, intermediateSize, lw.mlpInterBias, mlpInter);
        geluInPlace(mlpInter);
        matmulXWtPlusB(mlpInter, seqLen, intermediateSize, lw.mlpOutWeight, hiddenSize, lw.mlpOutBias, mlpOut);

        // Residual + LayerNorm
        for (int i = 0; i < seqLen * hiddenSize; i++) x[i] = x[i] + mlpOut[i];
        layerNormInPlace(x, seqLen, hiddenSize, lw.outLnGamma, lw.outLnBeta, layerNormEps);
    }

    /**
     * Berechnet {@code out[M,N] = x[M,K] · W[N,K]^T + bias[N]}.
     * W ist im PyTorch-Layout {@code [out_features, in_features]} gespeichert.
     */
    private static void matmulXWtPlusB(float[] x, int M, int K,
                                       float[] w, int N,
                                       float[] bias, float[] out) {
        for (int m = 0; m < M; m++) {
            int xi = m * K;
            int oi = m * N;
            for (int n = 0; n < N; n++) {
                int wi = n * K;
                float sum = 0f;
                for (int kk = 0; kk < K; kk++) {
                    sum += x[xi + kk] * w[wi + kk];
                }
                out[oi + n] = bias != null ? sum + bias[n] : sum;
            }
        }
    }

    /**
     * BERT-LayerNorm über die letzte Dimension, in-place.
     */
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

    /**
     * Exakte GELU (BERT-Standard): 0.5 · x · (1 + erf(x / √2)).
     */
    private static void geluInPlace(float[] x) {
        for (int i = 0; i < x.length; i++) {
            float xi = x[i];
            x[i] = (float) (0.5 * xi * (1.0 + erf(xi / Math.sqrt(2.0))));
        }
    }

    /**
     * Abramowitz & Stegun 7.1.26 – relative Fehler &lt; 1.5e-7.
     */
    private static double erf(double xRaw) {
        double sign = xRaw < 0 ? -1.0 : 1.0;
        double x = Math.abs(xRaw);
        double t = 1.0 / (1.0 + 0.3275911 * x);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
        return sign * y;
    }

    /**
     * Multi-Head Scaled Dot-Product Attention.
     * <p>
     * Tensoren werden im flachen {@code [seq * H]}-Layout interpretiert.
     * Heads sind interleaved über die letzte Dimension:
     * {@code x[t][h*D + d] = head h, dim d an Token t}.
     */
    private static void multiHeadAttention(float[] q, float[] k, float[] v,
                                           int[] attentionMask, int seqLen,
                                           int hiddenSize, int numHeads, int headDim,
                                           float[] out, float[] scoresBuf) {
        double scale = 1.0 / Math.sqrt(headDim);
        // Per Head
        for (int h = 0; h < numHeads; h++) {
            int headOffset = h * headDim;

            // 1) scores = Q · K^T / √d, dann Masking
            for (int i = 0; i < seqLen; i++) {
                int qBase = i * hiddenSize + headOffset;
                for (int j = 0; j < seqLen; j++) {
                    int kBase = j * hiddenSize + headOffset;
                    double sum = 0.0;
                    for (int d = 0; d < headDim; d++) {
                        sum += q[qBase + d] * k[kBase + d];
                    }
                    double score = sum * scale;
                    if (attentionMask[j] == 0) score = -1e9;
                    scoresBuf[i * seqLen + j] = (float) score;
                }
            }

            // 2) Softmax pro Zeile
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

            // 3) attn = probs · V
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

