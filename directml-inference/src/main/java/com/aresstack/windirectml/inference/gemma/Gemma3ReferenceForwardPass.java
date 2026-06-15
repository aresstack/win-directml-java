package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;

import java.util.Objects;

/**
 * Device-free CPU reference forward pass for Gemma 3 (parity oracle for the WARP runtime). Implements
 * the full Gemma layer: zero-centered RMSNorm, QK-norm, dual-theta RoPE, GQA attention with a
 * sliding-window mask on local layers, the GeGLU/GELU-tanh MLP, sandwich norms, and the tied LM head.
 *
 * <p>Not optimized and not the product path — it materializes everything in {@code float[]} and is
 * intended for small/synthetic configs and gated real-model parity checks.</p>
 */
public final class Gemma3ReferenceForwardPass {

    private final Gemma3Config config;
    private final Gemma3ReferenceWeights weights;

    public Gemma3ReferenceForwardPass(Gemma3ReferenceWeights weights) {
        this.weights = Objects.requireNonNull(weights, "weights");
        this.config = weights.config();
    }

    /** Run a full prefill over {@code tokenIds} and return the vocab-sized logits for the last position. */
    public float[] logitsForLastToken(int[] tokenIds) {
        if (tokenIds == null || tokenIds.length == 0) {
            throw new IllegalArgumentException("tokenIds must not be empty");
        }
        int s = tokenIds.length;
        int h = config.hiddenSize();
        float eps = (float) config.rmsNormEps();

        // 1. Embedding lookup + scaling by sqrt(hidden).
        float[][] hidden = new float[s][h];
        float embScale = (float) config.embeddingScale();
        for (int t = 0; t < s; t++) {
            int id = tokenIds[t];
            if (id < 0 || id >= config.vocabSize()) {
                throw new IllegalArgumentException("token id out of range: " + id);
            }
            int base = id * h;
            for (int i = 0; i < h; i++) {
                hidden[t][i] = weights.embedTokens[base + i] * embScale;
            }
        }

        for (int layer = 0; layer < config.numHiddenLayers(); layer++) {
            applyLayer(hidden, layer, eps);
        }

        // Final norm on the last position, then tied LM head.
        float[] last = hidden[s - 1].clone();
        Gemma3ReferenceMath.rmsNormZeroCentered(last, weights.finalNorm, eps);
        int vocab = config.vocabSize();
        float[] logits = new float[vocab];
        float[] lmHead = weights.lmHead();
        for (int o = 0; o < vocab; o++) {
            logits[o] = dot(lmHead, o * h, last, 0, h);
        }
        return logits;
    }

    /** Greedy next token id for {@code tokenIds}. */
    public int nextToken(int[] tokenIds) {
        return DecoderOnlyMath.argmax(logitsForLastToken(tokenIds));
    }

    /**
     * Run a single transformer layer in place over {@code state} ({@code [seqLen][hidden]}) — the
     * device-free parity oracle for {@link Gemma3WarpLayer} (GEMMA-WARP-8). Uses the configured
     * {@code rms_norm_eps}.
     */
    public void runLayer(float[][] state, int layer) {
        applyLayer(state, layer, (float) config.rmsNormEps());
    }

    private void applyLayer(float[][] hidden, int layer, float eps) {
        Gemma3ReferenceWeights.Layer w = weights.layers[layer];
        int s = hidden.length;
        int h = config.hiddenSize();
        int nH = config.numAttentionHeads();
        int nKV = config.numKeyValueHeads();
        int d = config.headDim();
        int attnDim = config.attentionDim();
        int kvDim = config.keyValueDim();
        double theta = config.ropeThetaForLayer(layer);
        boolean full = config.isFullAttentionLayer(layer);
        int window = config.slidingWindow();
        float attnScale = (float) config.attentionScale();
        int groupsPerKv = nH / nKV;

        // q/k/v projections + QK-norm + RoPE for every position.
        float[][] q = new float[s][attnDim];
        float[][] k = new float[s][kvDim];
        float[][] v = new float[s][kvDim];
        for (int t = 0; t < s; t++) {
            float[] normed = hidden[t].clone();
            Gemma3ReferenceMath.rmsNormZeroCentered(normed, w.inputLayerNorm, eps);
            matvec(w.qProj, normed, q[t], attnDim, h);
            matvec(w.kProj, normed, k[t], kvDim, h);
            matvec(w.vProj, normed, v[t], kvDim, h);
            for (int head = 0; head < nH; head++) {
                Gemma3ReferenceMath.rmsNormZeroCentered(q[t], head * d, d, w.qNorm, eps);
                Gemma3ReferenceMath.applyRopeHalf(q[t], head * d, d, t, theta);
            }
            for (int head = 0; head < nKV; head++) {
                Gemma3ReferenceMath.rmsNormZeroCentered(k[t], head * d, d, w.kNorm, eps);
                Gemma3ReferenceMath.applyRopeHalf(k[t], head * d, d, t, theta);
            }
        }

        // Attention -> oProj -> post-attention norm -> residual.
        for (int t = 0; t < s; t++) {
            float[] attnOut = new float[attnDim];
            int firstValid = full ? 0 : Math.max(0, t - window + 1);
            float[] scores = new float[t + 1];
            for (int head = 0; head < nH; head++) {
                int kvHead = head / groupsPerKv;
                int qBase = head * d;
                int kvBase = kvHead * d;
                float max = Float.NEGATIVE_INFINITY;
                for (int j = firstValid; j <= t; j++) {
                    float sc = dot(q[t], qBase, k[j], kvBase, d) * attnScale;
                    scores[j] = sc;
                    if (sc > max) {
                        max = sc;
                    }
                }
                double sum = 0;
                for (int j = firstValid; j <= t; j++) {
                    double e = Math.exp(scores[j] - max);
                    scores[j] = (float) e;
                    sum += e;
                }
                float invSum = (float) (1.0 / sum);
                for (int j = firstValid; j <= t; j++) {
                    float a = scores[j] * invSum;
                    int vBase = kvHead * d;
                    for (int c = 0; c < d; c++) {
                        attnOut[qBase + c] += a * v[j][vBase + c];
                    }
                }
            }
            float[] attnProj = new float[h];
            matvec(w.oProj, attnOut, attnProj, h, attnDim);
            Gemma3ReferenceMath.rmsNormZeroCentered(attnProj, w.postAttentionLayerNorm, eps);
            for (int i = 0; i < h; i++) {
                hidden[t][i] += attnProj[i];
            }
        }

        // GeGLU MLP with pre/post feedforward sandwich norms + residual.
        int inter = config.intermediateSize();
        for (int t = 0; t < s; t++) {
            float[] ff = hidden[t].clone();
            Gemma3ReferenceMath.rmsNormZeroCentered(ff, w.preFeedforwardLayerNorm, eps);
            float[] gate = new float[inter];
            float[] up = new float[inter];
            matvec(w.gateProj, ff, gate, inter, h);
            matvec(w.upProj, ff, up, inter, h);
            Gemma3ReferenceMath.geluTanhInPlace(gate);
            Gemma3ReferenceMath.multiplyInPlace(gate, up);
            float[] down = new float[h];
            matvec(w.downProj, gate, down, h, inter);
            Gemma3ReferenceMath.rmsNormZeroCentered(down, w.postFeedforwardLayerNorm, eps);
            for (int i = 0; i < h; i++) {
                hidden[t][i] += down[i];
            }
        }
    }

    /** y[o] = sum_i W[o*in + i] * x[i], for o in [0,out). */
    private static void matvec(float[] w, float[] x, float[] y, int out, int in) {
        for (int o = 0; o < out; o++) {
            y[o] = dot(w, o * in, x, 0, in);
        }
    }

    private static float dot(float[] a, int aOff, float[] b, int bOff, int len) {
        float sum = 0;
        for (int i = 0; i < len; i++) {
            sum += a[aOff + i] * b[bOff + i];
        }
        return sum;
    }
}
