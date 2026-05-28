package com.aresstack.windirectml.inference.qwen;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * CPU-only decoder runtime for Qwen2.5-Coder-Instruct.
 *
 * <p>Implements the full Qwen2 decoder stack:
 * <ol>
 *   <li>Token embedding lookup</li>
 *   <li>Per-layer: RMSNorm → Q/K/V projection → RoPE → GQA causal attention
 *       with KV-cache → O projection → residual → RMSNorm → SwiGLU MLP → residual</li>
 *   <li>Final RMSNorm → LM head logits</li>
 * </ol>
 *
 * <h2>Differences from Phi-3 runtime</h2>
 * <ul>
 *   <li><b>GQA</b>: Qwen 0.5B has 14 Q heads and 2 KV heads (7:1 ratio).
 *       Phi-3 has equal Q/KV heads (no grouping). The attention loop maps each
 *       Q head to its corresponding KV head via contiguous groups.</li>
 *   <li><b>Separate gate/up projections</b>: Qwen uses separate gate_proj and
 *       up_proj weight matrices. Phi-3 uses a fused gate_up_proj matrix.</li>
 *   <li><b>No activation scales</b>: Qwen does not use the per-projection
 *       activation scales that Phi-3 AWQ quantization requires.</li>
 *   <li><b>RoPE theta</b>: Qwen uses rope_theta=1,000,000 for long-context;
 *       the runtime computes RoPE on-the-fly from config rather than using a
 *       pre-computed cos/sin cache from the ONNX model.</li>
 *   <li><b>CPU-only</b>: No GPU/DirectML path in this initial implementation.</li>
 * </ul>
 *
 * <p>Constraints:
 * <ul>
 *   <li>Greedy decoding only (no sampling/beam search)</li>
 *   <li>Single-batch (batch_size=1)</li>
 *   <li>CPU-only (no DirectML acceleration)</li>
 * </ul>
 */
public final class Qwen2Runtime {

    private static final Logger log = LoggerFactory.getLogger(Qwen2Runtime.class);

    private final Qwen2Config config;
    private final Qwen2Weights weights;
    private final QwenTokenizer tokenizer;
    private final int qHeadsPerKvHead;

    // ── KV Cache (per-head layout for cache locality) ────────────────────
    // Layout: [layer][head][pos * headDim]
    private final float[][][] kvCacheK;
    private final float[][][] kvCacheV;
    private int cachedSeqLen;

    // ── Pre-computed RoPE cos/sin tables ─────────────────────────────────
    private final float[] ropeCosBuf;  // [maxPos * halfDim]
    private final float[] ropeSinBuf;  // [maxPos * halfDim]

    // ── Pre-allocated decode buffers (seqLen=1) ──────────────────────────
    private final float[] decBuf;         // [hidden] — current hidden state
    private final float[] decNormed;      // [hidden] — normed hidden state
    private final float[] decQ;           // [qSize]
    private final float[] decK;           // [kvSize]
    private final float[] decV;           // [kvSize]
    private final float[] decAttnOut;     // [qSize]
    private final float[] decOProj;       // [hidden]
    private final float[] decResidual;    // [hidden]
    private final float[] decPostNorm;    // [hidden]
    private final float[] decGate;        // [intermediateSize]
    private final float[] decUp;          // [intermediateSize]
    private final float[] decMlpAct;      // [intermediateSize]
    private final float[] decDown;        // [hidden]
    private final float[] decScores;      // [maxPos] — attention scores
    private final float[] decLogits;      // [vocabSize]

    // ── Generation quality parameters ────────────────────────────────────
    private float repetitionPenalty = 1.2f;

    // ── Profiling ────────────────────────────────────────────────────────
    private long profProjNs;
    private long profAttnNs;
    private long profNormNs;
    private long profActNs;
    private long profLmHeadNs;
    private long profPrefillNs;
    private int profSteps;
    private String lastProfile;

    /**
     * Construct a CPU-only Qwen2 runtime.
     *
     * @param config    model configuration
     * @param weights   loaded model weights
     * @param tokenizer Qwen BPE tokenizer
     */
    public Qwen2Runtime(Qwen2Config config, Qwen2Weights weights, QwenTokenizer tokenizer) {
        this.config = config;
        this.weights = weights;
        this.tokenizer = tokenizer;

        int numLayers = config.numHiddenLayers();
        int numHeads = config.numAttentionHeads();
        int kvHeads = config.numKeyValueHeads();
        if (numHeads % kvHeads != 0) {
            throw new IllegalArgumentException(
                    "numAttentionHeads must be divisible by numKeyValueHeads for GQA mapping");
        }
        this.qHeadsPerKvHead = numHeads / kvHeads;
        this.kvCacheK = new float[numLayers][kvHeads][];
        this.kvCacheV = new float[numLayers][kvHeads][];
        this.cachedSeqLen = 0;

        int hidden = config.hiddenSize();
        int qSize = config.qSize();
        int kvSize = config.kvSize();
        int intermediate = config.intermediateSize();
        int maxPos = config.maxPositionEmbeddings();

        // Pre-allocate decode buffers
        decBuf      = new float[hidden];
        decNormed   = new float[hidden];
        decQ        = new float[qSize];
        decK        = new float[kvSize];
        decV        = new float[kvSize];
        decAttnOut  = new float[qSize];
        decOProj    = new float[hidden];
        decResidual = new float[hidden];
        decPostNorm = new float[hidden];
        decGate     = new float[intermediate];
        decUp       = new float[intermediate];
        decMlpAct   = new float[intermediate];
        decDown     = new float[hidden];
        decScores   = new float[maxPos];
        decLogits   = new float[config.vocabSize()];

        // Pre-compute RoPE tables
        int halfDim = config.headDim() / 2;
        // Limit pre-computed RoPE table to 4096 positions to cap startup memory
        // (~2 MB for headDim=64). Positions beyond this are computed on-the-fly.
        int ropeMaxPos = Math.min(maxPos, 4096);
        ropeCosBuf = new float[ropeMaxPos * halfDim];
        ropeSinBuf = new float[ropeMaxPos * halfDim];
        double theta = config.ropeTheta();
        for (int pos = 0; pos < ropeMaxPos; pos++) {
            for (int i = 0; i < halfDim; i++) {
                double freq = 1.0 / Math.pow(theta, (2.0 * i) / config.headDim());
                double angle = pos * freq;
                ropeCosBuf[pos * halfDim + i] = (float) Math.cos(angle);
                ropeSinBuf[pos * halfDim + i] = (float) Math.sin(angle);
            }
        }

        log.info("Qwen2Runtime: CPU-only mode, {} layers, {} heads ({}KV), headDim={}, GQA ratio={}:1",
                numLayers, numHeads, kvHeads, config.headDim(), qHeadsPerKvHead);
    }

    // ── Streaming callback ────────────────────────────────────────────────

    @FunctionalInterface
    public interface TokenConsumer {
        void onToken(int tokenId, String textSoFar, String delta);
    }

    // ── Public API ───────────────────────────────────────────────────────

    /**
     * Generate tokens greedily from a prompt (non-streaming).
     *
     * @param prompt    text prompt (should be formatted with ChatML template)
     * @param maxTokens maximum number of tokens to generate
     * @return generated text (excluding the prompt)
     */
    public String generate(String prompt, int maxTokens) {
        return generateStreaming(prompt, maxTokens, null);
    }

    /**
     * Generate tokens greedily with a per-token streaming callback.
     *
     * @param prompt    text prompt (should be formatted with ChatML template)
     * @param maxTokens maximum number of tokens to generate (hard ceiling)
     * @param consumer  optional callback invoked after each token
     * @return generated text (excluding the prompt)
     */
    public String generateStreaming(String prompt, int maxTokens, TokenConsumer consumer) {
        int[] inputIds = tokenizer.encode(prompt);

        resetProfile();
        resetCache();

        // ── Prefill ──────────────────────────────────────────────────
        long t0 = System.nanoTime();
        float[] logits = prefill(inputIds);
        profPrefillNs = System.nanoTime() - t0;

        log.info("Prefill: {} tokens in {} ms", inputIds.length, String.format("%.1f", profPrefillNs / 1e6));

        // ── Decode loop ──────────────────────────────────────────────
        List<Integer> generatedIds = new ArrayList<>();
        String previousText = "";

        for (int step = 0; step < maxTokens; step++) {

            // Repetition penalty
            if (repetitionPenalty > 1.0f && !generatedIds.isEmpty()) {
                applyRepetitionPenalty(logits, generatedIds);
            }

            int nextToken = argmax(logits);

            if (QwenStopTokenPolicy.shouldStop(nextToken)) {
                break;
            }

            generatedIds.add(nextToken);

            // Decode accumulated IDs to text
            String fullText = tokenizer.decode(
                    generatedIds.stream().mapToInt(Integer::intValue).toArray(), true);
            String delta = fullText.substring(previousText.length());
            previousText = fullText;

            if (consumer != null) {
                consumer.onToken(nextToken, fullText, delta);
            }

            // Decode next token
            logits = decodeSingleToken(nextToken);
            profSteps++;
        }

        lastProfile = buildProfileSummary();
        log.debug("Profile: {}", lastProfile);

        return previousText;
    }

    public void setRepetitionPenalty(float penalty) {
        this.repetitionPenalty = Math.max(1.0f, penalty);
    }

    public String getLastProfile() {
        return lastProfile;
    }

    public void resetCache() {
        cachedSeqLen = 0;
        for (float[][] layer : kvCacheK) Arrays.fill(layer, null);
        for (float[][] layer : kvCacheV) Arrays.fill(layer, null);
    }

    // ── Prefill (process all prompt tokens) ──────────────────────────────

    private float[] prefill(int[] tokenIds) {
        int seqLen = tokenIds.length;
        int hidden = config.hiddenSize();

        // Embedding lookup
        float[] hiddenStates = new float[seqLen * hidden];
        for (int s = 0; s < seqLen; s++) {
            int tokenId = tokenIds[s];
            if (tokenId >= 0 && tokenId < config.vocabSize()) {
                System.arraycopy(weights.embedTokens, tokenId * hidden, hiddenStates, s * hidden, hidden);
            }
        }

        // Process each layer
        int totalLayers = config.numHiddenLayers();
        long layerStart = System.nanoTime();
        for (int l = 0; l < totalLayers; l++) {
            hiddenStates = processLayerPrefill(l, hiddenStates, seqLen, 0);
            if (l == 0 || (l + 1) % 4 == 0 || l == totalLayers - 1) {
                long elapsed = (System.nanoTime() - layerStart) / 1_000_000L;
                log.info("Prefill layer {}/{} done ({} ms elapsed, seqLen={})",
                        l + 1, totalLayers, elapsed, seqLen);
            }
        }

        // Final norm + logits (only for last position)
        float[] lastHidden = new float[hidden];
        System.arraycopy(hiddenStates, (seqLen - 1) * hidden, lastHidden, 0, hidden);
        rmsNorm(lastHidden, weights.finalNormWeight, config.rmsNormEps());

        float[] logits = new float[config.vocabSize()];
        Arrays.fill(logits, 0);
        weights.lmHead.matvec(lastHidden, logits);

        cachedSeqLen = seqLen;
        return logits;
    }

    // ── Single-token decode (uses pre-allocated buffers) ─────────────────

    private float[] decodeSingleToken(int tokenId) {
        int hidden = config.hiddenSize();
        int pos = cachedSeqLen;

        // Embedding lookup → decBuf
        if (tokenId >= 0 && tokenId < config.vocabSize()) {
            System.arraycopy(weights.embedTokens, tokenId * hidden, decBuf, 0, hidden);
        } else {
            Arrays.fill(decBuf, 0);
        }

        // Process each layer
        for (int l = 0; l < config.numHiddenLayers(); l++) {
            processLayerDecode(l, decBuf, pos);
        }

        // Final norm
        long t0 = System.nanoTime();
        rmsNorm(decBuf, weights.finalNormWeight, config.rmsNormEps());
        profNormNs += System.nanoTime() - t0;

        // LM head → decLogits
        t0 = System.nanoTime();
        Arrays.fill(decLogits, 0);
        weights.lmHead.matvec(decBuf, decLogits);
        profLmHeadNs += System.nanoTime() - t0;

        cachedSeqLen = pos + 1;
        return decLogits;
    }

    // ── Per-layer decode (zero-allocation hot path) ──────────────────────

    private void processLayerDecode(int layerIdx, float[] hiddenIo, int pos) {
        int hidden = config.hiddenSize();
        int numHeads = config.numAttentionHeads();
        int headDim = config.headDim();
        int kvHeads = config.numKeyValueHeads();
        int qSize = config.qSize();
        int kvSize = config.kvSize();
        Qwen2Weights.LayerWeights lw = weights.layers[layerIdx];
        long t0;

        // ── Pre-attention RMSNorm → decNormed ────────────────────────
        t0 = System.nanoTime();
        System.arraycopy(hiddenIo, 0, decNormed, 0, hidden);
        rmsNorm(decNormed, lw.inputNormWeight(), config.rmsNormEps());
        profNormNs += System.nanoTime() - t0;

        // ── Q/K/V Projections ────────────────────────────────────────
        t0 = System.nanoTime();
        Arrays.fill(decQ, 0);
        Arrays.fill(decK, 0);
        Arrays.fill(decV, 0);
        lw.qProj().matvec(decNormed, decQ);
        lw.kProj().matvec(decNormed, decK);
        lw.vProj().matvec(decNormed, decV);
        // Apply attention biases if present
        if (lw.qBias() != null) addBias(decQ, lw.qBias());
        if (lw.kBias() != null) addBias(decK, lw.kBias());
        if (lw.vBias() != null) addBias(decV, lw.vBias());
        profProjNs += System.nanoTime() - t0;

        // ── RoPE ─────────────────────────────────────────────────────
        t0 = System.nanoTime();
        for (int h = 0; h < numHeads; h++) {
            applyRoPE(decQ, h * headDim, headDim, pos);
        }
        for (int h = 0; h < kvHeads; h++) {
            applyRoPE(decK, h * headDim, headDim, pos);
        }
        profNormNs += System.nanoTime() - t0;

        // ── KV Cache update ──────────────────────────────────────────
        ensureKvLayerCapacity(layerIdx, pos + 1);
        for (int h = 0; h < kvHeads; h++) {
            System.arraycopy(decK, h * headDim, kvCacheK[layerIdx][h], pos * headDim, headDim);
            System.arraycopy(decV, h * headDim, kvCacheV[layerIdx][h], pos * headDim, headDim);
        }

        // ── Grouped-Query Attention ──────────────────────────────────
        t0 = System.nanoTime();
        Arrays.fill(decAttnOut, 0, qSize, 0.0f);
        float scale = (float) (1.0 / Math.sqrt(headDim));

        for (int h = 0; h < numHeads; h++) {
            int kvH = kvHeadForQueryHead(h);
            int qOff = h * headDim;
            float[] kHead = kvCacheK[layerIdx][kvH];
            float[] vHead = kvCacheV[layerIdx][kvH];

            // Q·K dot products
            for (int p = 0; p <= pos; p++) {
                int kOff = p * headDim;
                float dot = 0;
                for (int d = 0; d < headDim; d++) {
                    dot += decQ[qOff + d] * kHead[kOff + d];
                }
                decScores[p] = dot * scale;
            }

            // Softmax
            softmax(decScores, pos + 1);

            // Weighted sum of V
            int outOff = h * headDim;
            for (int p = 0; p <= pos; p++) {
                float w = decScores[p];
                if (w < 1e-8f) continue;
                int vOff = p * headDim;
                for (int d = 0; d < headDim; d++) {
                    decAttnOut[outOff + d] += w * vHead[vOff + d];
                }
            }
        }
        profAttnNs += System.nanoTime() - t0;

        // ── O projection → decOProj ──────────────────────────────────
        t0 = System.nanoTime();
        Arrays.fill(decOProj, 0);
        lw.oProj().matvec(decAttnOut, decOProj);
        profProjNs += System.nanoTime() - t0;

        // ── Residual 1 → decResidual ─────────────────────────────────
        for (int i = 0; i < hidden; i++) {
            decResidual[i] = hiddenIo[i] + decOProj[i];
        }

        // ── Post-attention RMSNorm → decPostNorm ─────────────────────
        t0 = System.nanoTime();
        System.arraycopy(decResidual, 0, decPostNorm, 0, hidden);
        rmsNorm(decPostNorm, lw.postNormWeight(), config.rmsNormEps());
        profNormNs += System.nanoTime() - t0;

        // ── MLP: gate_proj + up_proj → SwiGLU → down_proj ────────────
        t0 = System.nanoTime();
        Arrays.fill(decGate, 0);
        Arrays.fill(decUp, 0);
        lw.gateProj().matvec(decPostNorm, decGate);
        lw.upProj().matvec(decPostNorm, decUp);
        profProjNs += System.nanoTime() - t0;

        // SwiGLU activation: silu(gate) * up, where silu(x) = x * sigmoid(x)
        t0 = System.nanoTime();
        int intermediate = config.intermediateSize();
        for (int i = 0; i < intermediate; i++) {
            float gate = decGate[i];
            float sigmoid = 1.0f / (1.0f + (float) Math.exp(-gate));
            decMlpAct[i] = (gate * sigmoid) * decUp[i];
        }
        profActNs += System.nanoTime() - t0;

        // down_proj
        t0 = System.nanoTime();
        Arrays.fill(decDown, 0);
        lw.downProj().matvec(decMlpAct, decDown);
        profProjNs += System.nanoTime() - t0;

        // ── Residual 2 → hiddenIo (output, in-place) ────────────────
        for (int i = 0; i < hidden; i++) {
            hiddenIo[i] = decResidual[i] + decDown[i];
        }
    }

    // ── Prefill layer processing ─────────────────────────────────────────

    private float[] processLayerPrefill(int layerIdx, float[] input, int seqLen, int startPos) {
        int hidden = config.hiddenSize();
        int numHeads = config.numAttentionHeads();
        int headDim = config.headDim();
        int kvHeads = config.numKeyValueHeads();
        int qSize = config.qSize();
        int kvSize = config.kvSize();
        Qwen2Weights.LayerWeights lw = weights.layers[layerIdx];

        // ── Pre-attention RMSNorm ────────────────────────────────────
        float[] normed = new float[seqLen * hidden];
        for (int s = 0; s < seqLen; s++) {
            float[] row = new float[hidden];
            System.arraycopy(input, s * hidden, row, 0, hidden);
            rmsNorm(row, lw.inputNormWeight(), config.rmsNormEps());
            System.arraycopy(row, 0, normed, s * hidden, hidden);
        }

        // ── Q/K/V Projections ────────────────────────────────────────
        float[] q = new float[seqLen * qSize];
        float[] k = new float[seqLen * kvSize];
        float[] v = new float[seqLen * kvSize];
        lw.qProj().matmul(normed, q, seqLen);
        lw.kProj().matmul(normed, k, seqLen);
        lw.vProj().matmul(normed, v, seqLen);
        // Apply attention biases if present (broadcast across seq positions)
        if (lw.qBias() != null) addBiasBatched(q, lw.qBias(), seqLen);
        if (lw.kBias() != null) addBiasBatched(k, lw.kBias(), seqLen);
        if (lw.vBias() != null) addBiasBatched(v, lw.vBias(), seqLen);

        // ── RoPE ─────────────────────────────────────────────────────
        for (int s = 0; s < seqLen; s++) {
            int pos = startPos + s;
            for (int h = 0; h < numHeads; h++) {
                applyRoPE(q, s * qSize + h * headDim, headDim, pos);
            }
            for (int h = 0; h < kvHeads; h++) {
                applyRoPE(k, s * kvSize + h * headDim, headDim, pos);
            }
        }

        // ── KV Cache update ──────────────────────────────────────────
        ensureKvLayerCapacity(layerIdx, startPos + seqLen);
        for (int s = 0; s < seqLen; s++) {
            for (int h = 0; h < kvHeads; h++) {
                System.arraycopy(k, s * kvSize + h * headDim,
                        kvCacheK[layerIdx][h], (startPos + s) * headDim, headDim);
                System.arraycopy(v, s * kvSize + h * headDim,
                        kvCacheV[layerIdx][h], (startPos + s) * headDim, headDim);
            }
        }

        // ── Causal Self-Attention (GQA) ──────────────────────────────
        float[] attnOut = new float[seqLen * qSize];
        float scale = (float) (1.0 / Math.sqrt(headDim));

        for (int s = 0; s < seqLen; s++) {
            int queryPos = startPos + s;
            for (int h = 0; h < numHeads; h++) {
                int kvH = kvHeadForQueryHead(h);
                int qOff = s * qSize + h * headDim;
                float[] kHead = kvCacheK[layerIdx][kvH];
                float[] vHead = kvCacheV[layerIdx][kvH];

                float[] scores = new float[queryPos + 1];
                for (int p = 0; p <= queryPos; p++) {
                    int kOff = p * headDim;
                    float dot = 0;
                    for (int d = 0; d < headDim; d++) {
                        dot += q[qOff + d] * kHead[kOff + d];
                    }
                    scores[p] = dot * scale;
                }

                softmax(scores, scores.length);

                int outOff = s * qSize + h * headDim;
                for (int p = 0; p <= queryPos; p++) {
                    float w = scores[p];
                    int vOff = p * headDim;
                    for (int d = 0; d < headDim; d++) {
                        attnOut[outOff + d] += w * vHead[vOff + d];
                    }
                }
            }
        }

        // ── O projection ─────────────────────────────────────────────
        float[] oProjOut = new float[seqLen * hidden];
        lw.oProj().matmul(attnOut, oProjOut, seqLen);

        // ── Residual 1 + Post-attention RMSNorm ──────────────────────
        float[] residual1 = new float[seqLen * hidden];
        for (int i = 0; i < seqLen * hidden; i++) {
            residual1[i] = input[i] + oProjOut[i];
        }

        float[] postNormed = new float[seqLen * hidden];
        for (int s = 0; s < seqLen; s++) {
            float[] row = new float[hidden];
            System.arraycopy(residual1, s * hidden, row, 0, hidden);
            rmsNorm(row, lw.postNormWeight(), config.rmsNormEps());
            System.arraycopy(row, 0, postNormed, s * hidden, hidden);
        }

        // ── MLP: gate + up → SwiGLU → down ──────────────────────────
        int intermediate = config.intermediateSize();
        float[] gate = new float[seqLen * intermediate];
        float[] up = new float[seqLen * intermediate];
        lw.gateProj().matmul(postNormed, gate, seqLen);
        lw.upProj().matmul(postNormed, up, seqLen);

        // SwiGLU: silu(gate) * up
        float[] mlpActivation = new float[seqLen * intermediate];
        for (int s = 0; s < seqLen; s++) {
            int off = s * intermediate;
            for (int i = 0; i < intermediate; i++) {
                float g = gate[off + i];
                float sigmoid = 1.0f / (1.0f + (float) Math.exp(-g));
                mlpActivation[off + i] = (g * sigmoid) * up[off + i];
            }
        }

        // down_proj
        float[] downOut = new float[seqLen * hidden];
        lw.downProj().matmul(mlpActivation, downOut, seqLen);

        // ── Residual 2 ──────────────────────────────────────────────
        float[] output = new float[seqLen * hidden];
        for (int i = 0; i < seqLen * hidden; i++) {
            output[i] = residual1[i] + downOut[i];
        }

        return output;
    }

    // ── KV cache management ──────────────────────────────────────────────

    private void ensureKvLayerCapacity(int layer, int requiredPositions) {
        int headDim = config.headDim();
        int kvHeads = config.numKeyValueHeads();
        int requiredLength = requiredPositions * headDim;
        for (int h = 0; h < kvHeads; h++) {
            if (kvCacheK[layer][h] == null || kvCacheK[layer][h].length < requiredLength) {
                int newCapacity = Math.max(requiredPositions, 64);
                if (kvCacheK[layer][h] != null) {
                    newCapacity = Math.max(newCapacity, kvCacheK[layer][h].length / headDim * 2);
                }
                int newLength = newCapacity * headDim;
                float[] newK = new float[newLength];
                float[] newV = new float[newLength];
                if (kvCacheK[layer][h] != null) {
                    System.arraycopy(kvCacheK[layer][h], 0, newK, 0, kvCacheK[layer][h].length);
                    System.arraycopy(kvCacheV[layer][h], 0, newV, 0, kvCacheV[layer][h].length);
                }
                kvCacheK[layer][h] = newK;
                kvCacheV[layer][h] = newV;
            }
        }
    }

    static int kvHeadForQueryHead(int queryHead, int qHeadsPerKvHead, int numKvHeads) {
        int kvHead = queryHead / qHeadsPerKvHead;
        if (kvHead >= numKvHeads) {
            throw new IllegalArgumentException("queryHead out of range for configured GQA mapping: " + queryHead);
        }
        return kvHead;
    }

    private int kvHeadForQueryHead(int queryHead) {
        return kvHeadForQueryHead(queryHead, qHeadsPerKvHead, config.numKeyValueHeads());
    }

    // ── Math utilities ───────────────────────────────────────────────────

    static void rmsNorm(float[] x, float[] weight, float eps) {
        float sumSq = 0;
        for (float v : x) sumSq += v * v;
        float rms = (float) (1.0 / Math.sqrt(sumSq / x.length + eps));
        for (int i = 0; i < x.length; i++) {
            x[i] = x[i] * rms * weight[i];
        }
    }

    /** Add bias vector to a single-position output: out[i] += bias[i]. */
    private static void addBias(float[] out, float[] bias) {
        for (int i = 0; i < bias.length; i++) {
            out[i] += bias[i];
        }
    }

    /** Add bias vector to batched output: out[s * dim + i] += bias[i] for each sequence position. */
    private static void addBiasBatched(float[] out, float[] bias, int seqLen) {
        int dim = bias.length;
        for (int s = 0; s < seqLen; s++) {
            int offset = s * dim;
            for (int i = 0; i < dim; i++) {
                out[offset + i] += bias[i];
            }
        }
    }

    private void applyRoPE(float[] vec, int offset, int dim, int pos) {
        int halfDim = dim / 2;
        int ropeMaxPos = ropeCosBuf.length / halfDim;
        if (pos < ropeMaxPos) {
            // Use pre-computed table
            for (int i = 0; i < halfDim; i++) {
                float cos = ropeCosBuf[pos * halfDim + i];
                float sin = ropeSinBuf[pos * halfDim + i];
                float x0 = vec[offset + i];
                float x1 = vec[offset + halfDim + i];
                vec[offset + i] = x0 * cos - x1 * sin;
                vec[offset + halfDim + i] = x0 * sin + x1 * cos;
            }
        } else {
            // Compute on the fly for positions beyond pre-computed table
            double theta = config.ropeTheta();
            for (int i = 0; i < halfDim; i++) {
                double freq = 1.0 / Math.pow(theta, (2.0 * i) / dim);
                double angle = pos * freq;
                float cos = (float) Math.cos(angle);
                float sin = (float) Math.sin(angle);
                float x0 = vec[offset + i];
                float x1 = vec[offset + halfDim + i];
                vec[offset + i] = x0 * cos - x1 * sin;
                vec[offset + halfDim + i] = x0 * sin + x1 * cos;
            }
        }
    }

    static void softmax(float[] x, int len) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < len; i++) if (x[i] > max) max = x[i];
        float sum = 0;
        for (int i = 0; i < len; i++) {
            x[i] = (float) Math.exp(x[i] - max);
            sum += x[i];
        }
        float invSum = 1.0f / sum;
        for (int i = 0; i < len; i++) x[i] *= invSum;
    }

    static int argmax(float[] logits) {
        int maxIdx = 0;
        float maxVal = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    // ── Repetition penalty ───────────────────────────────────────────────

    private void applyRepetitionPenalty(float[] logits, List<Integer> generatedIds) {
        boolean[] seen = new boolean[logits.length];
        for (int id : generatedIds) {
            if (id >= 0 && id < logits.length && !seen[id]) {
                seen[id] = true;
                if (logits[id] > 0) {
                    logits[id] /= repetitionPenalty;
                } else {
                    logits[id] *= repetitionPenalty;
                }
            }
        }
    }

    // ── Profiling ────────────────────────────────────────────────────────

    private void resetProfile() {
        profProjNs = profAttnNs = profNormNs = profActNs = profLmHeadNs = profPrefillNs = 0;
        profSteps = 0;
        lastProfile = null;
    }

    private String buildProfileSummary() {
        if (profSteps == 0) return "[No tokens generated]";
        long totalDecode = profProjNs + profAttnNs + profNormNs + profActNs + profLmHeadNs;
        double totalMs = totalDecode / 1e6;
        double perToken = totalMs / profSteps;
        double pctDivisor = totalDecode > 0 ? totalDecode : 1;

        return String.format(
                "[Qwen2 Decode Profile] %d tokens, %.1f ms total, %.1f ms/token%n"
                + "  Prefill:       %.1f ms%n"
                + "  Projections:   %.1f ms avg (%.0f%%)%n"
                + "  Attention:     %.1f ms avg (%.0f%%)%n"
                + "  Norms+RoPE:    %.1f ms avg (%.0f%%)%n"
                + "  SwiGLU:        %.1f ms avg (%.0f%%)%n"
                + "  LM head:       %.1f ms avg (%.0f%%)",
                profSteps, totalMs, perToken,
                profPrefillNs / 1e6,
                profProjNs / 1e6 / profSteps, 100.0 * profProjNs / pctDivisor,
                profAttnNs / 1e6 / profSteps, 100.0 * profAttnNs / pctDivisor,
                profNormNs / 1e6 / profSteps, 100.0 * profNormNs / pctDivisor,
                profActNs / 1e6 / profSteps, 100.0 * profActNs / pctDivisor,
                profLmHeadNs / 1e6 / profSteps, 100.0 * profLmHeadNs / pctDivisor
        );
    }
}
