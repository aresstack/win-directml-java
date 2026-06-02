package com.aresstack.windirectml.inference.phi3;

import com.aresstack.windirectml.inference.phi3.Phi3Weights.LayerWeights;
import com.aresstack.windirectml.windows.MatMulNBitsKernel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Decoder runtime for Phi-3-mini-4k-instruct.
 *
 * <p>Implements the full Phi-3 decoder stack:
 * <ol>
 *   <li>Token embedding lookup</li>
 *   <li>Per-layer: RMSNorm, Q/K/V projection, RoPE, causal attention with KV-cache,
 *       O projection, residual, RMSNorm, SwiGLU MLP, residual</li>
 *   <li>Final RMSNorm + LM head logits</li>
 * </ol>
 *
 * <p>V1 supports two execution modes:
 * <ul>
 *   <li><b>CPU-only</b>: all matrix multiplications on CPU (default)</li>
 *   <li><b>GPU-accelerated</b>: quantized projections dispatched to DirectML
 *       via {@link Phi3GpuKernels}; attention, norms, and activations remain on CPU</li>
 * </ul>
 *
 * <p><b>V1.3 KV-cache optimization</b>: the KV cache uses a <b>per-head layout</b>
 * ({@code [layer][head][pos * headDim]}) instead of the original interleaved layout
 * ({@code [layer][pos * allHeads * headDim]}). This reduces the memory stride between
 * consecutive positions from {@code numKvHeads * headDim * 4 = 12 KB} down to
 * {@code headDim * 4 = 384 bytes}, enabling hardware prefetching and dramatically
 * improving CPU cache utilization during attention (measured: ~8–10× faster attention).
 *
 * <p><b>V1.4 parallel attention</b>: attention heads are processed in parallel
 * via {@code IntStream.parallel()} when seqLen exceeds a threshold (32).
 * Each head writes to a non-overlapping slice of the output buffer, making
 * the operation data-race-free. Per-head score buffers are pre-allocated
 * to avoid heap allocation in the hot path. Expected speedup: 4–8× on
 * the attention bottleneck (with 8+ CPU cores).
 *
 * <p><b>V1.4 ILP dot product</b>: 4-accumulator dot product in the inner
 * attention loop for better instruction-level parallelism. Breaks the
 * dependency chain on a single FP accumulator, enabling out-of-order
 * execution to pipeline 4 independent FMA streams.
 *
 * <p>Constraints:
 * <ul>
 *   <li>Greedy decoding only (no sampling)</li>
 *   <li>Single-batch (batch_size=1)</li>
 * </ul>
 */
public final class Phi3Runtime {

    private static final Logger log = LoggerFactory.getLogger(Phi3Runtime.class);

    private final Phi3Config config;
    private final Phi3Weights weights;
    private final Phi3Tokenizer tokenizer;
    private final Phi3GpuKernels gpuKernels;
    private final Phi3GpuPipeline gpuPipeline;  // V2.0 shared pipeline (nullable)

    private final float[][][] kvCacheK;   // [layer][head][pos * headDim] — per-head K cache
    private final float[][][] kvCacheV;   // [layer][head][pos * headDim] — per-head V cache
    private int cachedSeqLen;

    // ── KV-cache token tracking for incremental prefill ──────────────────
    private int[] cachedTokenIds;  // token IDs currently represented in KV cache

    // ── Pre-allocated decode buffers (seqLen=1, reused across layers) ─────
    private final float[] decBuf;         // general-purpose [hidden]
    private final float[] decQKV;         // [3 * hidden] — fused QKV result (GPU)
    private final float[] decQ;           // [hidden]
    private final float[] decK;           // [hidden]
    private final float[] decV;           // [hidden]
    private final float[] decAttnOut;     // [hidden]
    private final float[] decOProj;       // [hidden]
    private final float[] decResidual;    // [hidden]
    private final float[] decPostNorm;    // [hidden]
    private final float[] decGateUp;      // [intermediateSize * 2]
    private final float[] decMlpAct;      // [intermediateSize]
    private final float[] decDown;        // [hidden]
    private final float[] decScores;      // [maxPositionEmbeddings] — attention scores (sequential path)
    private final float[] decLogits;      // [vocabSize]

    // ── Per-head score buffers for parallel attention (V1.4) ─────────────
    // Each head needs its own score buffer to avoid data races in parallel.
    // Layout: flat [numHeads * maxPos] — decScoresPool[h * maxPos + p]
    private final float[] decScoresPool;
    private final int maxPos;

    /**
     * Sequence length threshold above which attention heads are processed
     * in parallel. Below this threshold, thread scheduling overhead exceeds
     * the parallelism benefit.
     */
    private static final int PARALLEL_ATTENTION_THRESHOLD = 32;

    // ── Decode profiling (accumulated per generation, reset on each call) ─
    private long profGpuProjNs;
    private long profCpuAttnNs;
    private long profCpuNormNs;
    private long profCpuActNs;
    private long profLmHeadNs;
    private long profTokenDecNs;
    private long profPrefillNs;
    private int profSteps;
    private int profPrefillTokens;
    private String lastProfile;

    /**
     * CPU-only constructor (backward compatible).
     */
    public Phi3Runtime(Phi3Config config, Phi3Weights weights, Phi3Tokenizer tokenizer) {
        this(config, weights, tokenizer, null, null);
    }

    /**
     * @param gpuKernels optional GPU kernel pool — {@code null} for CPU-only mode
     */
    public Phi3Runtime(Phi3Config config, Phi3Weights weights, Phi3Tokenizer tokenizer,
                       Phi3GpuKernels gpuKernels) {
        this(config, weights, tokenizer, gpuKernels, null);
    }

    /**
     * Full constructor with V2.0 GPU pipeline support.
     *
     * @param gpuKernels  optional GPU kernel pool (fallback path)
     * @param gpuPipeline optional V2.0 shared pipeline (preferred if non-null)
     */
    public Phi3Runtime(Phi3Config config, Phi3Weights weights, Phi3Tokenizer tokenizer,
                       Phi3GpuKernels gpuKernels, Phi3GpuPipeline gpuPipeline) {
        this.config = config;
        this.weights = weights;
        this.tokenizer = tokenizer;
        this.gpuKernels = gpuKernels;
        this.gpuPipeline = gpuPipeline;
        this.kvCacheK = new float[config.numHiddenLayers()][config.numKeyValueHeads()][];
        this.kvCacheV = new float[config.numHiddenLayers()][config.numKeyValueHeads()][];
        this.cachedSeqLen = 0;
        this.cachedTokenIds = new int[0];

        // Pre-allocate decode buffers (seqLen=1)
        int hidden = config.hiddenSize();
        int interX2 = config.intermediateSize() * 2;
        int inter = config.intermediateSize();
        int maxPos = config.maxPositionEmbeddings();
        this.maxPos = maxPos;

        decBuf = new float[hidden];
        decQKV = new float[hidden * 3];
        decQ = new float[hidden];
        decK = new float[hidden];
        decV = new float[hidden];
        decAttnOut = new float[hidden];
        decOProj = new float[hidden];
        decResidual = new float[hidden];
        decPostNorm = new float[hidden];
        decGateUp = new float[interX2];
        decMlpAct = new float[inter];
        decDown = new float[hidden];
        decScores = new float[maxPos];
        decLogits = new float[config.vocabSize()];
        decScoresPool = new float[config.numAttentionHeads() * maxPos];

        if (gpuPipeline != null) {
            log.info("Phi3Runtime: V2.0 pipeline mode — shared cmd infra, MLP batch");
        } else if (gpuKernels != null) {
            log.info("Phi3Runtime: GPU mode — {}/{} layers on GPU, lmHead={}",
                    gpuKernels.getGpuLayers(), config.numHiddenLayers(), gpuKernels.hasLmHead());
        } else {
            log.info("Phi3Runtime: CPU-only mode");
        }
    }

    // ── Streaming callback ────────────────────────────────────────────────

    /**
     * Callback for token-by-token streaming during generation.
     */
    @FunctionalInterface
    public interface TokenConsumer {
        /**
         * Called after each generated token.
         *
         * @param tokenId   the token ID just generated
         * @param textSoFar full decoded text of all tokens generated so far
         * @param delta     new text fragment appended in this step
         */
        void onToken(int tokenId, String textSoFar, String delta);
    }

    // ── Public API ───────────────────────────────────────────────────────

    /**
     * Generate tokens greedily from a prompt (non-streaming).
     *
     * @param prompt    text prompt
     * @param maxTokens maximum number of tokens to generate
     * @return generated text (excluding the prompt)
     */
    public String generate(String prompt, int maxTokens) {
        return generateStreaming(prompt, maxTokens, null);
    }

    /**
     * Returns the formatted profile string from the last generation, or null.
     */
    public String getLastProfile() {
        return lastProfile;
    }

    // ── Generation quality parameters ────────────────────────────────────

    /**
     * Penalty factor for tokens already in the generated output (1.0 = off).
     * Positive logits are divided by the penalty, negative logits multiplied.
     * Typical values: 1.1 – 1.3.  This is a standard quality improvement for
     * greedy decoding — it reduces monotonous repetitions.
     */
    private float repetitionPenalty = 1.2f;

    public void setRepetitionPenalty(float penalty) {
        this.repetitionPenalty = Math.max(1.0f, penalty);
    }

    /**
     * Generate tokens greedily with a per-token streaming callback.
     * <p>
     * Token IDs are accumulated and decoded as a full sequence after each
     * step so that SentencePiece inter-token spaces are preserved correctly.
     * <p>
     * Includes a <b>repetition penalty</b> — a standard quality improvement
     * for greedy decoding that reduces monotonous repetitions by penalising
     * logits of already-generated tokens.
     *
     * @param prompt    text prompt
     * @param maxTokens maximum number of tokens to generate (hard ceiling)
     * @param consumer  optional callback invoked after each token (may be {@code null})
     * @return generated text (excluding the prompt)
     */
    public String generateStreaming(String prompt, int maxTokens, TokenConsumer consumer) {
        int[] inputIds = tokenizer.encode(prompt);

        resetProfile();

        // ── Incremental prefill: find common prefix with cached tokens ──
        int commonPrefix = 0;
        if (cachedTokenIds != null) {
            int limit = Math.min(cachedTokenIds.length, inputIds.length);
            while (commonPrefix < limit && cachedTokenIds[commonPrefix] == inputIds[commonPrefix]) {
                commonPrefix++;
            }
        }

        // Trim cache to the common prefix (invalidate anything after it)
        cachedSeqLen = commonPrefix;

        int newTokenCount = inputIds.length - commonPrefix;
        log.info("Prompt: {} tokens (cached={}, new={})", inputIds.length, commonPrefix, newTokenCount);

        // Prefill only the NEW suffix tokens
        long t0 = System.nanoTime();
        float[] logits;
        if (newTokenCount > 0) {
            int[] suffixIds = new int[newTokenCount];
            System.arraycopy(inputIds, commonPrefix, suffixIds, 0, newTokenCount);
            logits = prefill(suffixIds, commonPrefix);
        } else {
            // Entire prompt is already cached — recompute logits for last position only
            logits = recomputeLastLogits(inputIds[inputIds.length - 1]);
        }
        profPrefillNs = System.nanoTime() - t0;
        profPrefillTokens = newTokenCount;

        // Track what's in the cache now (= the full prompt)
        cachedTokenIds = Arrays.copyOf(inputIds, inputIds.length);

        // ── Decode loop ──────────────────────────────────────────────
        List<Integer> generatedIds = new ArrayList<>();
        String previousText = "";

        for (int step = 0; step < maxTokens; step++) {

            // ── Repetition penalty (quality improvement for greedy decoding) ─
            if (repetitionPenalty > 1.0f && !generatedIds.isEmpty()) {
                applyRepetitionPenalty(logits, generatedIds);
            }

            int nextToken = argmax(logits);

            if (tokenizer.isEos(nextToken)) {
                break;
            }

            generatedIds.add(nextToken);

            // Decode full sequence to preserve SentencePiece spaces
            long td0 = System.nanoTime();
            String fullText = tokenizer.decode(
                    generatedIds.stream().mapToInt(Integer::intValue).toArray());
            String delta = fullText.substring(previousText.length());
            previousText = fullText;
            profTokenDecNs += System.nanoTime() - td0;

            if (consumer != null) {
                consumer.onToken(nextToken, fullText, delta);
            }

            // Decode: process single token using pre-allocated buffers
            logits = decodeFast(nextToken);
            profSteps++;

            // Extend cached token tracking
            cachedTokenIds = Arrays.copyOf(cachedTokenIds, cachedTokenIds.length + 1);
            cachedTokenIds[cachedTokenIds.length - 1] = nextToken;
        }

        lastProfile = buildProfileSummary();
        log.debug("Profile: {}", lastProfile);

        return previousText;
    }

    /**
     * Apply repetition penalty to logits for all tokens that have already
     * been generated.  Positive logits are divided by the penalty, negative
     * logits are multiplied — so repeated tokens are always pushed down.
     */
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
        profGpuProjNs = profCpuAttnNs = profCpuNormNs = profCpuActNs = 0;
        profLmHeadNs = profTokenDecNs = profPrefillNs = 0;
        profSteps = 0;
        profPrefillTokens = 0;
        lastProfile = null;
    }

    private String buildProfileSummary() {
        if (profSteps == 0) return "[No tokens generated]";
        long totalDecode = profGpuProjNs + profCpuAttnNs + profCpuNormNs + profCpuActNs + profLmHeadNs;
        double totalMs = totalDecode / 1e6;
        double perToken = totalMs / profSteps;
        double pctDivisor = totalDecode > 0 ? totalDecode : 1;

        int gpuLayers = (gpuKernels != null) ? gpuKernels.getGpuLayers() : 0;
        boolean mlpBatch = gpuPipeline != null && gpuPipeline.isMlpBatchEnabled();

        int gpuSubmissions;
        String pipelineTag;
        if (mlpBatch) {
            int subsPerLayer = 2;  // QKV + MLP_batch
            gpuSubmissions = gpuLayers * subsPerLayer + ((gpuKernels != null && gpuKernels.hasLmHead()) ? 1 : 0);
            pipelineTag = "V2.0 MLP-batch";
        } else {
            int subsPerLayer = 4;
            gpuSubmissions = gpuLayers * subsPerLayer + ((gpuKernels != null && gpuKernels.hasLmHead()) ? 1 : 0);
            pipelineTag = gpuPipeline != null ? "V2.0 pipeline" : "V1.x per-kernel";
        }

        StringBuilder sb = new StringBuilder();
        sb.append(String.format("[Decode Profile] %d tokens, %.1f ms total, %.1f ms/token%n", profSteps, totalMs, perToken));
        sb.append(String.format("  Prefill:         %.1f ms (%d new tokens, %d cached)%n",
                profPrefillNs / 1e6, profPrefillTokens, cachedSeqLen - profSteps - profPrefillTokens));
        sb.append(String.format("  GPU projections: %.1f ms avg (%.0f%%) [%d subs/tok, %s]%n",
                profGpuProjNs / 1e6 / profSteps, 100.0 * profGpuProjNs / pctDivisor, gpuSubmissions, pipelineTag));
        sb.append(String.format("  CPU attention:   %.1f ms avg (%.0f%%) [%d heads, parallel=%s]%n",
                profCpuAttnNs / 1e6 / profSteps, 100.0 * profCpuAttnNs / pctDivisor,
                config.numAttentionHeads(),
                Runtime.getRuntime().availableProcessors() > 1 ? "yes (cores=" + Runtime.getRuntime().availableProcessors() + ")" : "no"));
        sb.append(String.format("  CPU norms+RoPE:  %.1f ms avg (%.0f%%)%n",
                profCpuNormNs / 1e6 / profSteps, 100.0 * profCpuNormNs / pctDivisor));
        sb.append(String.format("  CPU SwiGLU:      %.1f ms avg (%.0f%%)%n",
                profCpuActNs / 1e6 / profSteps, 100.0 * profCpuActNs / pctDivisor));
        sb.append(String.format("  LM head:         %.1f ms avg (%.0f%%)%n",
                profLmHeadNs / 1e6 / profSteps, 100.0 * profLmHeadNs / pctDivisor));
        sb.append(String.format("  Token decode:    %.1f ms avg%n", profTokenDecNs / 1e6 / profSteps));
        return sb.toString();
    }

    public void resetCache() {
        cachedSeqLen = 0;
        cachedTokenIds = new int[0];
        for (float[][] layer : kvCacheK) Arrays.fill(layer, null);
        for (float[][] layer : kvCacheV) Arrays.fill(layer, null);
    }

    /**
     * Ensure KV cache arrays for all heads in a layer have enough capacity.
     * Uses 2× growth factor for amortized O(1) expansion.
     */
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

    // ── Prefill (supports incremental via startPos) ──────────────────────

    /**
     * Process multiple tokens (prefill phase) starting at a given position.
     * Positions [0..startPos-1] are assumed already in the KV cache.
     */
    private float[] prefill(int[] tokenIds, int startPos) {
        int seqLen = tokenIds.length;
        int hidden = config.hiddenSize();

        // Embedding lookup
        float[] hidden_states = new float[seqLen * hidden];
        for (int s = 0; s < seqLen; s++) {
            System.arraycopy(weights.embedTokens, tokenIds[s] * hidden, hidden_states, s * hidden, hidden);
        }

        // Process each layer
        for (int l = 0; l < config.numHiddenLayers(); l++) {
            hidden_states = processLayer(l, hidden_states, seqLen, startPos);
        }

        // Final norm + logits (only for last position)
        float[] lastHidden = new float[hidden];
        System.arraycopy(hidden_states, (seqLen - 1) * hidden, lastHidden, 0, hidden);
        rmsNorm(lastHidden, weights.finalNormWeight, config.rmsNormEps());

        float[] logits = new float[config.vocabSize()];
        lmHeadMatvec(lastHidden, logits);

        cachedSeqLen = startPos + seqLen;
        return logits;
    }

    /**
     * Recompute logits for the last cached position (when full prompt is cached).
     * This is a lightweight single-token forward pass.
     */
    private float[] recomputeLastLogits(int lastTokenId) {
        return decodeFast(lastTokenId);
    }

    // ── Fast single-token decode (pre-allocated buffers) ─────────────────

    /**
     * Process a single new token using pre-allocated buffers.
     * <p>
     * V2.0: per-layer path (65 submissions with MLP batch) or CPU-only.
     */
    private float[] decodeFast(int tokenId) {
        int hidden = config.hiddenSize();
        int pos = cachedSeqLen;

        // Embedding lookup → decBuf
        System.arraycopy(weights.embedTokens, tokenId * hidden, decBuf, 0, hidden);

        // Process each layer
        for (int l = 0; l < config.numHiddenLayers(); l++) {
            processLayerDecode(l, decBuf, pos);
        }

        // Final norm
        long t0 = System.nanoTime();
        rmsNorm(decBuf, weights.finalNormWeight, config.rmsNormEps());
        profCpuNormNs += System.nanoTime() - t0;

        // LM head → decLogits
        t0 = System.nanoTime();
        lmHeadMatvec(decBuf, decLogits);
        profLmHeadNs += System.nanoTime() - t0;

        cachedSeqLen = pos + 1;
        return decLogits;
    }

    // ── LM head ──────────────────────────────────────────────────────────

    private void lmHeadMatvec(float[] x, float[] logits) {
        if (gpuPipeline != null && gpuPipeline.hasLmHead()) {
            gpuPipeline.lmHead(x, logits);
        } else if (gpuKernels != null && gpuKernels.hasLmHead()) {
            gpuKernels.lmHead().matvec(x, logits);
        } else {
            Arrays.fill(logits, 0);  // zero before matvec (accumulates via +=)
            weights.lmHead.matvec(x, logits);
        }
    }

    // ── Zero-alloc single-token layer processing ─────────────────────────

    /**
     * Process one decoder layer for a single token (decode phase).
     * Uses pre-allocated buffers — zero heap allocation in the hot path.
     * <p>
     * Buffer assignments:
     * <pre>
     *   decOProj  → normed input (temporary, reused later for O projection)
     *   decQ      → Q projection result
     *   decK      → K projection result
     *   decV      → V projection result (temporary, stored into KV cache)
     *   decScores → attention scores [pos+1]
     *   decAttnOut→ attention output
     *   decOProj  → O projection result (reuses normed buffer)
     *   decResidual → residual1
     *   decPostNorm → post-attention norm
     *   decGateUp → gate+up projection
     *   decMlpAct → SwiGLU activation
     *   decDown   → down projection
     * </pre>
     *
     * @param layerIdx  layer index
     * @param hidden_io input/output hidden state [hidden] — modified in-place
     * @param pos       absolute position for this token
     */
    private void processLayerDecode(int layerIdx, float[] hidden_io, int pos) {
        int hidden = config.hiddenSize();
        int numHeads = config.numAttentionHeads();
        int headDim = config.headDim();
        int kvHeads = config.numKeyValueHeads();
        LayerWeights lw = weights.layers[layerIdx];
        boolean pipelineLayer = gpuPipeline != null && gpuPipeline.hasLayer(layerIdx);
        boolean gpuLayer = pipelineLayer || (gpuKernels != null && gpuKernels.hasLayer(layerIdx));
        long t0;

        // ── Pre-attention RMSNorm → decOProj (as "normed" temp) ──────
        t0 = System.nanoTime();
        System.arraycopy(hidden_io, 0, decOProj, 0, hidden);
        rmsNorm(decOProj, lw.inputNormWeight(), config.rmsNormEps());
        // decOProj now holds the normed input for Q/K/V projections
        profCpuNormNs += System.nanoTime() - t0;

        // ── Q/K/V Projections (all read from decOProj = normed) ──────
        // IMPORTANT: QuantizedWeight.matvec() ACCUMULATES (y[n] += sum),
        // so output buffers must be zeroed before each CPU-path call.
        t0 = System.nanoTime();
        if (pipelineLayer) {
            // V2.0: QKV via shared pipeline (shared allocator/cmdlist/fence)
            gpuPipeline.qkvFused(layerIdx, decOProj, decQKV);
            System.arraycopy(decQKV, 0, decQ, 0, hidden);
            System.arraycopy(decQKV, hidden, decK, 0, hidden);
            System.arraycopy(decQKV, 2 * hidden, decV, 0, hidden);
        } else if (gpuLayer) {
            // V1.x fallback: per-kernel submission
            gpuKernels.qkvFused(layerIdx).matvec(decOProj, decQKV);
            System.arraycopy(decQKV, 0, decQ, 0, hidden);
            System.arraycopy(decQKV, hidden, decK, 0, hidden);
            System.arraycopy(decQKV, 2 * hidden, decV, 0, hidden);
        } else {
            Arrays.fill(decQ, 0);
            Arrays.fill(decK, 0);
            Arrays.fill(decV, 0);
            lw.qProj().matvec(decOProj, decQ);
            lw.kProj().matvec(decOProj, decK);
            lw.vProj().matvec(decOProj, decV);
        }
        profGpuProjNs += System.nanoTime() - t0;

        // ── RoPE ─────────────────────────────────────────────────────
        t0 = System.nanoTime();
        for (int h = 0; h < numHeads; h++) {
            applyRoPE(decQ, h * headDim, headDim, pos);
        }
        for (int h = 0; h < kvHeads; h++) {
            applyRoPE(decK, h * headDim, headDim, pos);
        }
        profCpuNormNs += System.nanoTime() - t0;

        // ── KV Cache update (per-head layout for cache locality) ─────
        // Old layout: stride = kvHeads*headDim = 12KB between positions (thrashes CPU cache)
        // New layout: stride = headDim = 384 bytes between positions (sequential, prefetch-friendly)
        ensureKvLayerCapacity(layerIdx, pos + 1);
        for (int h = 0; h < kvHeads; h++) {
            System.arraycopy(decK, h * headDim, kvCacheK[layerIdx][h], pos * headDim, headDim);
            System.arraycopy(decV, h * headDim, kvCacheV[layerIdx][h], pos * headDim, headDim);
        }

        // ── Causal Self-Attention (V1.4: parallel heads + 4-acc dot product) ─
        t0 = System.nanoTime();
        Arrays.fill(decAttnOut, 0, hidden, 0.0f);
        float scale = (float) (1.0 / Math.sqrt(headDim));

        if (pos >= PARALLEL_ATTENTION_THRESHOLD) {
            // Parallel path: process 32 heads concurrently via ForkJoinPool.
            // Each head writes to a non-overlapping slice [h*headDim..(h+1)*headDim)
            // of decAttnOut — no data race. Per-head scores are stored in
            // decScoresPool[h*maxPos..h*maxPos+pos] to avoid heap allocation.
            final int posF = pos;
            final float scaleF = scale;
            IntStream.range(0, numHeads).parallel().forEach(h -> {
                int kvH = h % kvHeads;
                int qOff = h * headDim;
                float[] kHead = kvCacheK[layerIdx][kvH];
                float[] vHead = kvCacheV[layerIdx][kvH];
                int scoreBase = h * maxPos;

                // Q·K dot products with 4-accumulator ILP
                for (int p = 0; p <= posF; p++) {
                    int kOff = p * headDim;
                    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    int d = 0;
                    for (; d + 3 < headDim; d += 4) {
                        d0 += decQ[qOff + d] * kHead[kOff + d];
                        d1 += decQ[qOff + d + 1] * kHead[kOff + d + 1];
                        d2 += decQ[qOff + d + 2] * kHead[kOff + d + 2];
                        d3 += decQ[qOff + d + 3] * kHead[kOff + d + 3];
                    }
                    float dot = d0 + d1 + d2 + d3;
                    for (; d < headDim; d++) {
                        dot += decQ[qOff + d] * kHead[kOff + d];
                    }
                    decScoresPool[scoreBase + p] = dot * scaleF;
                }

                // In-place softmax over [scoreBase..scoreBase+pos]
                softmaxSlice(decScoresPool, scoreBase, posF + 1);

                // Weighted sum of V
                int outOff = h * headDim;
                for (int p = 0; p <= posF; p++) {
                    float w = decScoresPool[scoreBase + p];
                    if (w < 1e-8f) continue;  // skip negligible weights
                    int vOff = p * headDim;
                    for (int d = 0; d < headDim; d++) {
                        decAttnOut[outOff + d] += w * vHead[vOff + d];
                    }
                }
            });
        } else {
            // Sequential path for short sequences (avoids thread scheduling overhead)
            for (int h = 0; h < numHeads; h++) {
                int kvH = h % kvHeads;
                int qOff = h * headDim;
                float[] kHead = kvCacheK[layerIdx][kvH];
                float[] vHead = kvCacheV[layerIdx][kvH];

                // Q·K dot products with 4-accumulator ILP
                for (int p = 0; p <= pos; p++) {
                    int kOff = p * headDim;
                    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    int d = 0;
                    for (; d + 3 < headDim; d += 4) {
                        d0 += decQ[qOff + d] * kHead[kOff + d];
                        d1 += decQ[qOff + d + 1] * kHead[kOff + d + 1];
                        d2 += decQ[qOff + d + 2] * kHead[kOff + d + 2];
                        d3 += decQ[qOff + d + 3] * kHead[kOff + d + 3];
                    }
                    float dot = d0 + d1 + d2 + d3;
                    for (; d < headDim; d++) {
                        dot += decQ[qOff + d] * kHead[kOff + d];
                    }
                    decScores[p] = dot * scale;
                }

                softmax(decScores, pos + 1);

                int outOff = h * headDim;
                for (int p = 0; p <= pos; p++) {
                    int vOff = p * headDim;
                    float w = decScores[p];
                    for (int d = 0; d < headDim; d++) {
                        decAttnOut[outOff + d] += w * vHead[vOff + d];
                    }
                }
            }
        }
        profCpuAttnNs += System.nanoTime() - t0;

        // ── Activation scale (always CPU — fast element-wise) ─────────
        t0 = System.nanoTime();
        for (int i = 0; i < hidden; i++) {
            decAttnOut[i] *= lw.attnOutScale()[i];
        }
        profCpuNormNs += System.nanoTime() - t0;

        // ══════════════════════════════════════════════════════════════
        // V2.0 MLP BATCH: 7 GPU ops in 1 submission (65 subs/token)
        //   O_proj → add(residual) → RMSNorm → GateUp → SwiGLU → Down → add
        // ══════════════════════════════════════════════════════════════
        if (pipelineLayer && gpuPipeline.isMlpBatchEnabled()) {
            t0 = System.nanoTime();
            gpuPipeline.batchMlp(decAttnOut, hidden_io, hidden_io, layerIdx);
            profGpuProjNs += System.nanoTime() - t0;
            return;  // hidden_io updated in-place — layer done
        }

        // ══════════════════════════════════════════════════════════════
        // Fallback: per-kernel dispatch (V1.x path)
        // ══════════════════════════════════════════════════════════════

        // ── O projection → decOProj ──────────────────────────────────
        t0 = System.nanoTime();
        if (pipelineLayer) {
            gpuPipeline.oProj(layerIdx, decAttnOut, decOProj);
        } else if (gpuLayer) {
            gpuKernels.oProj(layerIdx).matvec(decAttnOut, decOProj);
        } else {
            Arrays.fill(decOProj, 0);
            lw.oProj().matvec(decAttnOut, decOProj);
        }
        profGpuProjNs += System.nanoTime() - t0;

        // ── Residual 1 → decResidual ─────────────────────────────────
        t0 = System.nanoTime();
        for (int i = 0; i < hidden; i++) {
            decResidual[i] = hidden_io[i] + decOProj[i];
        }

        // ── Post-attention RMSNorm → decPostNorm ─────────────────────
        System.arraycopy(decResidual, 0, decPostNorm, 0, hidden);
        rmsNorm(decPostNorm, lw.postNormWeight(), config.rmsNormEps());
        profCpuNormNs += System.nanoTime() - t0;

        // ── MLP: gate_up_proj → decGateUp ────────────────────────────
        t0 = System.nanoTime();
        if (pipelineLayer) {
            gpuPipeline.gateUpProj(layerIdx, decPostNorm, decGateUp);
        } else if (gpuLayer) {
            gpuKernels.gateUpProj(layerIdx).matvec(decPostNorm, decGateUp);
        } else {
            Arrays.fill(decGateUp, 0);
            lw.gateUpProj().matvec(decPostNorm, decGateUp);
        }
        profGpuProjNs += System.nanoTime() - t0;

        // ── SwiGLU activation + MLP scale (fused, V1.4) → decMlpAct ─
        t0 = System.nanoTime();
        int intermediate = config.intermediateSize();
        float[] mlpScale = lw.mlpOutScale();
        for (int i = 0; i < intermediate; i++) {
            float gate = decGateUp[i];
            float up = decGateUp[intermediate + i];
            float sigmoid = 1.0f / (1.0f + (float) Math.exp(-gate));
            decMlpAct[i] = up * gate * sigmoid * mlpScale[i];
        }
        profCpuActNs += System.nanoTime() - t0;

        // ── down_proj → decDown ──────────────────────────────────────
        t0 = System.nanoTime();
        if (pipelineLayer) {
            gpuPipeline.downProj(layerIdx, decMlpAct, decDown);
        } else if (gpuLayer) {
            gpuKernels.downProj(layerIdx).matvec(decMlpAct, decDown);
        } else {
            Arrays.fill(decDown, 0);
            lw.downProj().matvec(decMlpAct, decDown);
        }
        profGpuProjNs += System.nanoTime() - t0;

        // ── Residual 2 → hidden_io (output, in-place) ───────────────
        for (int i = 0; i < hidden; i++) {
            hidden_io[i] = decResidual[i] + decDown[i];
        }
    }

    // ── Prefill layer processing (allocates per call, used only during prefill) ──

    /**
     * Process one decoder layer for multiple tokens (prefill phase).
     */
    private float[] processLayer(int layerIdx, float[] input, int seqLen, int startPos) {
        int hidden = config.hiddenSize();
        int numHeads = config.numAttentionHeads();
        int headDim = config.headDim();
        int kvHeads = config.numKeyValueHeads();
        LayerWeights lw = weights.layers[layerIdx];
        boolean gpuLayer = gpuKernels != null && gpuKernels.hasLayer(layerIdx);

        // ── Pre-attention RMSNorm ────────────────────────────────────
        float[] normed = new float[seqLen * hidden];
        for (int s = 0; s < seqLen; s++) {
            float[] row = new float[hidden];
            System.arraycopy(input, s * hidden, row, 0, hidden);
            rmsNorm(row, lw.inputNormWeight(), config.rmsNormEps());
            System.arraycopy(row, 0, normed, s * hidden, hidden);
        }

        // ── Q/K/V Projections ────────────────────────────────────────
        float[] q = new float[seqLen * hidden];
        float[] k = new float[seqLen * hidden];
        float[] v = new float[seqLen * hidden];

        if (gpuLayer) {
            // Fused QKV: single GPU call per position → 3× fewer submissions
            for (int s = 0; s < seqLen; s++) {
                float[] row = new float[hidden];
                System.arraycopy(normed, s * hidden, row, 0, hidden);
                float[] qkvResult = gpuKernels.qkvFused(layerIdx).matvec(row);
                System.arraycopy(qkvResult, 0, q, s * hidden, hidden);
                System.arraycopy(qkvResult, hidden, k, s * hidden, hidden);
                System.arraycopy(qkvResult, 2 * hidden, v, s * hidden, hidden);
            }
        } else {
            lw.qProj().matmul(normed, q, seqLen);
            lw.kProj().matmul(normed, k, seqLen);
            lw.vProj().matmul(normed, v, seqLen);
        }

        // ── RoPE ─────────────────────────────────────────────────────
        for (int s = 0; s < seqLen; s++) {
            int pos = startPos + s;
            for (int h = 0; h < numHeads; h++) {
                applyRoPE(q, s * hidden + h * headDim, headDim, pos);
            }
            for (int h = 0; h < kvHeads; h++) {
                applyRoPE(k, s * hidden + h * headDim, headDim, pos);
            }
        }

        // ── KV Cache update (per-head layout for cache locality) ─────
        ensureKvLayerCapacity(layerIdx, startPos + seqLen);
        for (int s = 0; s < seqLen; s++) {
            for (int h = 0; h < kvHeads; h++) {
                System.arraycopy(k, s * hidden + h * headDim,
                        kvCacheK[layerIdx][h], (startPos + s) * headDim, headDim);
                System.arraycopy(v, s * hidden + h * headDim,
                        kvCacheV[layerIdx][h], (startPos + s) * headDim, headDim);
            }
        }

        // ── Causal Self-Attention (per-head KV cache for locality) ───
        float[] attnOut = new float[seqLen * hidden];
        float scale = (float) (1.0 / Math.sqrt(headDim));

        for (int s = 0; s < seqLen; s++) {
            int queryPos = startPos + s;
            for (int h = 0; h < numHeads; h++) {
                int kvH = h % kvHeads;
                int qOff = s * hidden + h * headDim;
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

                softmax(scores);

                int outOff = s * hidden + h * headDim;
                for (int p = 0; p <= queryPos; p++) {
                    int vOff = p * headDim;
                    float w = scores[p];
                    for (int d = 0; d < headDim; d++) {
                        attnOut[outOff + d] += w * vHead[vOff + d];
                    }
                }
            }
        }

        // ── Activation scale + O projection ──────────────────────────
        for (int s = 0; s < seqLen; s++) {
            for (int i = 0; i < hidden; i++) {
                attnOut[s * hidden + i] *= lw.attnOutScale()[i];
            }
        }

        float[] oProjOut = new float[seqLen * hidden];
        if (gpuLayer) {
            gpuMatmul(gpuKernels.oProj(layerIdx), attnOut, oProjOut, seqLen, hidden, hidden);
        } else {
            lw.oProj().matmul(attnOut, oProjOut, seqLen);
        }

        // ── Residual 1 + Post-attention RMSNorm ─────────────────────
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

        // ── MLP: gate_up_proj ────────────────────────────────────────
        int intermediateX2 = config.intermediateSize() * 2;
        float[] gateUp = new float[seqLen * intermediateX2];
        if (gpuLayer) {
            gpuMatmul(gpuKernels.gateUpProj(layerIdx), postNormed, gateUp,
                    seqLen, hidden, intermediateX2);
        } else {
            lw.gateUpProj().matmul(postNormed, gateUp, seqLen);
        }

        // ── SwiGLU activation ────────────────────────────────────────
        int intermediate = config.intermediateSize();
        float[] mlpActivation = new float[seqLen * intermediate];
        for (int s = 0; s < seqLen; s++) {
            int guOff = s * intermediateX2;
            int outOff = s * intermediate;
            for (int i = 0; i < intermediate; i++) {
                float gate = gateUp[guOff + i];
                float up = gateUp[guOff + intermediate + i];
                float sigmoid = 1.0f / (1.0f + (float) Math.exp(-gate));
                mlpActivation[outOff + i] = up * gate * sigmoid;
            }
        }

        // ── Activation scale + down_proj ─────────────────────────────
        for (int s = 0; s < seqLen; s++) {
            for (int i = 0; i < intermediate; i++) {
                mlpActivation[s * intermediate + i] *= lw.mlpOutScale()[i];
            }
        }

        float[] downOut = new float[seqLen * hidden];
        if (gpuLayer) {
            gpuMatmul(gpuKernels.downProj(layerIdx), mlpActivation, downOut,
                    seqLen, intermediate, hidden);
        } else {
            lw.downProj().matmul(mlpActivation, downOut, seqLen);
        }

        // ── Residual 2 ──────────────────────────────────────────────
        float[] output = new float[seqLen * hidden];
        for (int i = 0; i < seqLen * hidden; i++) {
            output[i] = residual1[i] + downOut[i];
        }

        return output;
    }

    // ── GPU matmul helper ─────────────────────────────────────────────────

    private static void gpuMatmul(MatMulNBitsKernel kernel,
                                  float[] input, float[] output,
                                  int seqLen, int K, int N) {
        for (int s = 0; s < seqLen; s++) {
            float[] row = new float[K];
            System.arraycopy(input, s * K, row, 0, K);
            float[] result = kernel.matvec(row);
            System.arraycopy(result, 0, output, s * N, N);
        }
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

    private void applyRoPE(float[] vec, int offset, int dim, int pos) {
        int halfDim = dim / 2;
        for (int i = 0; i < halfDim; i++) {
            float cos = weights.cosCache[pos * halfDim + i];
            float sin = weights.sinCache[pos * halfDim + i];
            float x0 = vec[offset + i];
            float x1 = vec[offset + halfDim + i];
            vec[offset + i] = x0 * cos - x1 * sin;
            vec[offset + halfDim + i] = x0 * sin + x1 * cos;
        }
    }

    /**
     * In-place softmax over the first {@code len} elements of the array.
     */
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

    /**
     * In-place softmax over a slice of a shared array.
     * Operates on {@code x[offset..offset+len-1]}.
     * Used by parallel attention to avoid per-head array allocation.
     */
    static void softmaxSlice(float[] x, int offset, int len) {
        float max = Float.NEGATIVE_INFINITY;
        int end = offset + len;
        for (int i = offset; i < end; i++) if (x[i] > max) max = x[i];
        float sum = 0;
        for (int i = offset; i < end; i++) {
            x[i] = (float) Math.exp(x[i] - max);
            sum += x[i];
        }
        float invSum = 1.0f / sum;
        for (int i = offset; i < end; i++) x[i] *= invSum;
    }

    /**
     * In-place softmax over the entire array.
     */
    static void softmax(float[] x) {
        softmax(x, x.length);
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
}
