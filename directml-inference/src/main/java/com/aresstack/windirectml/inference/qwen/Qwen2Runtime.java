package com.aresstack.windirectml.inference.qwen;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

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
 *   <li>Attention + norms on CPU; projections optionally on GPU via
 *       {@link QwenGpuKernels}</li>
 * </ul>
 */
public final class Qwen2Runtime {

    private static final Logger log = LoggerFactory.getLogger(Qwen2Runtime.class);

    private final Qwen2Config config;
    private final Qwen2Weights weights;
    private final QwenTokenizer tokenizer;
    private final int qHeadsPerKvHead;
    private final QwenGpuKernels gpuKernels;   // null if CPU-only
    private final QwenGpuPipeline gpuPipeline; // null if V1 or CPU-only
    private final boolean useGpuForDecode;     // false in HYBRID mode — GPU only for prefill

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
    private final float[] decQKV;         // [qSize + 2*kvSize] — fused QKV GPU output
    private final float[] decAttnOut;     // [qSize]
    private final float[] decOProj;       // [hidden]
    private final float[] decResidual;    // [hidden]
    private final float[] decPostNorm;    // [hidden]
    private final float[] decGate;        // [intermediateSize]  (CPU path)
    private final float[] decUp;          // [intermediateSize]  (CPU path)
    private final float[] decGateUp;      // [2*intermediateSize] (GPU path)
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
    // Per-stage prefill timing (Opt-A-2 diagnosis 2026-05-29). Aggregated over all 24 layers.
    private long profPrefillQkvBatchNs;
    private long profPrefillOProjBatchNs;
    private long profPrefillGuBatchNs;
    private long profPrefillDownBatchNs;
    private long profPrefillAttnNs;
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
        this(config, weights, tokenizer, null);
    }

    /**
     * Construct a Qwen2 runtime with optional GPU acceleration (V1 — per-kernel dispatches).
     *
     * @param config     model configuration
     * @param weights    loaded model weights
     * @param tokenizer  Qwen BPE tokenizer
     * @param gpuKernels optional DirectML GPU kernels; {@code null} = CPU-only
     */
    public Qwen2Runtime(Qwen2Config config, Qwen2Weights weights, QwenTokenizer tokenizer,
                        QwenGpuKernels gpuKernels) {
        this(config, weights, tokenizer, gpuKernels, null);
    }

    /**
     * Construct a Qwen2 runtime with V2.0 GPU pipeline (batched MLP, 48 fence-waits/token).
     *
     * @param config      model configuration
     * @param weights     loaded model weights
     * @param tokenizer   Qwen BPE tokenizer
     * @param gpuKernels  optional DirectML GPU kernels; {@code null} = CPU-only
     * @param gpuPipeline optional V2.0 GPU pipeline; {@code null} = fall back to V1 or CPU
     */
    public Qwen2Runtime(Qwen2Config config, Qwen2Weights weights, QwenTokenizer tokenizer,
                        QwenGpuKernels gpuKernels, QwenGpuPipeline gpuPipeline) {
        this(config, weights, tokenizer, gpuKernels, gpuPipeline, true);
    }

    /**
     * Construct a Qwen2 runtime with full backend control.
     *
     * @param config          model configuration
     * @param weights         loaded model weights
     * @param tokenizer       Qwen BPE tokenizer
     * @param gpuKernels      optional DirectML GPU kernels; {@code null} = CPU-only
     * @param gpuPipeline     optional V2.0 GPU pipeline; {@code null} = fall back to V1 or CPU
     * @param useGpuForDecode if {@code false}, GPU is used only for prefill;
     *                        per-token decode (and lm_head) stays on CPU. This is the
     *                        {@code HYBRID} backend mode and is recommended on Intel
     *                        iGPU hosts where the per-submission fence overhead
     *                        dominates per-token decode cost.
     */
    public Qwen2Runtime(Qwen2Config config, Qwen2Weights weights, QwenTokenizer tokenizer,
                        QwenGpuKernels gpuKernels, QwenGpuPipeline gpuPipeline,
                        boolean useGpuForDecode) {
        this.config = config;
        this.weights = weights;
        this.tokenizer = tokenizer;
        this.gpuKernels = gpuKernels;
        this.gpuPipeline = gpuPipeline;
        this.useGpuForDecode = useGpuForDecode;

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
        decBuf = new float[hidden];
        decNormed = new float[hidden];
        decQ = new float[qSize];
        decK = new float[kvSize];
        decV = new float[kvSize];
        decQKV = new float[qSize + 2 * kvSize];
        decAttnOut = new float[qSize];
        decOProj = new float[hidden];
        decResidual = new float[hidden];
        decPostNorm = new float[hidden];
        decGate = new float[intermediate];
        decUp = new float[intermediate];
        decGateUp = new float[intermediate * 2];
        decMlpAct = new float[intermediate];
        decDown = new float[hidden];
        decScores = new float[maxPos];
        decLogits = new float[config.vocabSize()];

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

        String modeStr;
        if (!useGpuForDecode && (gpuPipeline != null || gpuKernels != null)) {
            modeStr = "HYBRID (GPU prefill + CPU decode)";
        } else if (gpuPipeline != null) {
            modeStr = "GPU-V2(pipeline, " + gpuPipeline.hasLayer(config.numHiddenLayers() - 1)
                    + " layers, mlpBatch=" + gpuPipeline.isMlpBatchEnabled() + ")";
        } else if (gpuKernels != null) {
            modeStr = "GPU-V1(" + gpuKernels.getGpuLayers() + " layers)";
        } else {
            modeStr = "CPU-only";
        }
        log.info("Qwen2Runtime: {} mode, {} layers, {} heads ({}KV), headDim={}, GQA ratio={}:1",
                modeStr, numLayers, numHeads, kvHeads, config.headDim(), qHeadsPerKvHead);

        // ── One-shot diagnostic: SIMD / parallelism / heap ──────────────────────────────────
        // The CPU path's per-token cost is dominated by INT4 SIMD matvec
        // (Qwen2Weights.QuantizedWeight.matvec).  Logging these three numbers
        // here makes the next "why is decode/prefill slow?" investigation a
        // one-line answer instead of a multi-hour code review.
        Runtime rt = Runtime.getRuntime();
        long maxHeapMb = rt.maxMemory() / (1024L * 1024L);
        int avail = rt.availableProcessors();
        int fjp = ForkJoinPool.commonPool().getParallelism();
        log.info("Qwen2Runtime perf diag: SimdOps.enabled={}, availableProcessors={}, "
                        + "commonPool.parallelism={}, maxHeap={}MB, jvm={} {}",
                SimdOps.enabled(), avail, fjp, maxHeapMb,
                System.getProperty("java.vm.name"), System.getProperty("java.version"));
        if (!SimdOps.enabled()) {
            log.warn("Qwen2Runtime: SIMD path DISABLED — jdk.incubator.vector module not loaded. "
                    + "CPU matvec will be 4–16× slower than expected. Re-launch with "
                    + "`--add-modules=jdk.incubator.vector` (the Workbench launcher adds it "
                    + "automatically; direct `java -jar` invocations must add it explicitly).");
        }
        if (fjp < 2) {
            log.warn("Qwen2Runtime: ForkJoinPool parallelism={} — CPU prefill/decode will be "
                            + "single-threaded. On hardened/locked-down hosts check that the JVM is "
                            + "not restricted via `-Djava.util.concurrent.ForkJoinPool.common.parallelism`.",
                    fjp);
        }
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
        int[] generatedIds = new int[maxTokens];
        int generatedCount = 0;

        // Incremental UTF-8 byte accumulator for the streaming delta. The
        // previous implementation re-decoded `generatedIds.stream().toArray()`
        // every token — O(N^2) over the whole conversation. With a 200-token
        // output that meant ~20k redundant HashMap lookups in UNICODE_TO_BYTE
        // accumulated over the run. Now we decode each new token's bytes once
        // and append to a ByteArrayOutputStream — O(N) total. The byte buffer
        // also correctly resolves multi-byte UTF-8 codepoints that span two
        // tokens (the partial bytes carry across iterations).
        ByteArrayOutputStream decodedBytes = new ByteArrayOutputStream(256);
        String previousText = "";

        // Periodic profile log so the user can see WHERE time is going during
        // a long decode without waiting for the run to finish.
        final int profileLogEveryTokens = Integer.getInteger("qwen.profile.log.every", 16);
        long decodeWallStart = System.nanoTime();
        long lastProfileWall = decodeWallStart;

        for (int step = 0; step < maxTokens; step++) {

            // Repetition penalty (cheap, only touches the generatedCount unique ids)
            if (repetitionPenalty > 1.0f && generatedCount > 0) {
                applyRepetitionPenalty(logits, generatedIds, generatedCount);
            }

            int nextToken = argmax(logits);

            if (QwenStopTokenPolicy.shouldStop(nextToken)) {
                break;
            }

            generatedIds[generatedCount++] = nextToken;

            // Incremental decode: decode just this token's bytes and append.
            byte[] tokBytes = tokenizer.decode(new int[]{nextToken}, true)
                    .getBytes(StandardCharsets.UTF_8);
            decodedBytes.write(tokBytes, 0, tokBytes.length);
            String fullText = decodedBytes.toString(StandardCharsets.UTF_8);
            String delta = fullText.length() >= previousText.length()
                    ? fullText.substring(previousText.length())
                    : "";
            previousText = fullText;

            if (consumer != null) {
                consumer.onToken(nextToken, fullText, delta);
            }

            // Decode next token
            logits = decodeSingleToken(nextToken);
            profSteps++;

            // Periodic profile log — emit after every K decoded tokens so the
            // user sees real per-stage timing while the run is still going.
            if (profileLogEveryTokens > 0 && profSteps % profileLogEveryTokens == 0) {
                long now = System.nanoTime();
                double sinceLastSec = (now - lastProfileWall) / 1e9;
                double tokensPerSec = profileLogEveryTokens / Math.max(1e-9, sinceLastSec);
                lastProfileWall = now;
                log.info("Decode progress: {} tokens, last {} took {} s ({} tok/s) — profile so far:\n{}",
                        profSteps, profileLogEveryTokens,
                        String.format("%.2f", sinceLastSec),
                        String.format("%.2f", tokensPerSec),
                        buildProfileSummary());
            }
        }

        lastProfile = buildProfileSummary();
        double totalDecodeSec = (System.nanoTime() - decodeWallStart) / 1e9;
        log.info("Decode complete: {} tokens in {} s ({} tok/s)\n{}",
                profSteps,
                String.format("%.2f", totalDecodeSec),
                String.format("%.2f", profSteps / Math.max(1e-9, totalDecodeSec)),
                lastProfile);

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
        // Reset per-stage prefill counters (Opt-A-2 diagnosis).
        profPrefillQkvBatchNs = 0;
        profPrefillOProjBatchNs = 0;
        profPrefillGuBatchNs = 0;
        profPrefillDownBatchNs = 0;
        profPrefillAttnNs = 0;
        for (int l = 0; l < totalLayers; l++) {
            hiddenStates = processLayerPrefill(l, hiddenStates, seqLen, 0);
            if (l == 0 || (l + 1) % 4 == 0 || l == totalLayers - 1) {
                long elapsed = (System.nanoTime() - layerStart) / 1_000_000L;
                log.info("Prefill layer {}/{} done ({} ms elapsed, seqLen={})",
                        l + 1, totalLayers, elapsed, seqLen);
            }
        }
        // Per-stage prefill breakdown so we can see WHERE prefill time goes.
        // Opt-A-2 only touched FP32 batched shader (qkvFused + gateUpFused).
        // oProj + downProj run INT4 batched shader (unchanged).
        // cpuOther = (overall Prefill ms from outer log) - sumStages.
        long perStageSumMs = (profPrefillQkvBatchNs + profPrefillOProjBatchNs
                + profPrefillGuBatchNs + profPrefillDownBatchNs
                + profPrefillAttnNs) / 1_000_000L;
        log.info("Prefill per-stage ({} layers, seqLen={}): qkvBatch(FP32)={} ms, "
                        + "oProjBatch(INT4)={} ms, gateUpBatch(FP32)={} ms, downBatch(INT4)={} ms, "
                        + "attn(CPU)={} ms; sumStages={} ms",
                totalLayers, seqLen,
                profPrefillQkvBatchNs / 1_000_000L,
                profPrefillOProjBatchNs / 1_000_000L,
                profPrefillGuBatchNs / 1_000_000L,
                profPrefillDownBatchNs / 1_000_000L,
                profPrefillAttnNs / 1_000_000L,
                perStageSumMs);

        // Final norm + logits (only for last position)
        float[] lastHidden = new float[hidden];
        System.arraycopy(hiddenStates, (seqLen - 1) * hidden, lastHidden, 0, hidden);
        rmsNorm(lastHidden, weights.finalNormWeight, config.rmsNormEps());

        float[] logits = new float[config.vocabSize()];
        if (gpuPipeline != null && gpuPipeline.hasLmHead()) {
            gpuPipeline.lmHead(lastHidden, logits);
        } else if (gpuKernels != null && gpuKernels.hasLmHead()) {
            gpuKernels.lmHead().matvec(lastHidden, logits);
        } else {
            Arrays.fill(logits, 0);
            weights.lmHead.matvec(lastHidden, logits);
        }

        cachedSeqLen = seqLen;

        // ── Opt-B: lazy KV cache enable + upload ──
        // Try to enable the GPU-resident KV cache NOW — prefill batch scratch is
        // fully allocated, so the GPU memory peak is known. On Intel iGPU this
        // is the only timing that fits both batch scratch (~860 MB for seqLen=256)
        // AND the KV cache (~48 MB) into the available VRAM. If allocation fails,
        // tryEnableLazyGpuAttention logs a warning and decode keeps the CPU path.
        if (gpuPipeline != null) {
            gpuPipeline.tryEnableLazyGpuAttention();
        }
        if (gpuPipeline != null && gpuPipeline.isAttnGpuResidentEnabled()) {
            long upStart = System.nanoTime();
            int uploaded = 0;
            for (int l = 0; l < config.numHiddenLayers(); l++) {
                if (kvCacheK[l] == null || kvCacheK[l][0] == null) continue;
                try {
                    gpuPipeline.uploadKvCacheFromCpu(l, seqLen, kvCacheK[l], kvCacheV[l]);
                    uploaded++;
                } catch (Exception e) {
                    log.warn("Opt-B: KV cache upload failed for layer {} - falling back to CPU decode: {}",
                            l, e.getMessage());
                    return logits;
                }
            }
            log.info("Opt-B: uploaded prefill KV cache to GPU ({} layers, seqLen={}, {} ms)",
                    uploaded, seqLen, (System.nanoTime() - upStart) / 1_000_000L);

            // Opt-C: enable chained decode after KV cache upload
            gpuPipeline.tryEnableChainedDecode();
        }

        return logits;
    }

    // ── Single-token decode (uses pre-allocated buffers) ─────────────────

    /**
     * True if any GPU acceleration is configured AND enabled for the decode path.
     * In HYBRID mode this is false even when {@link #gpuPipeline} / {@link #gpuKernels}
     * are non-null — the GPU is then reserved for prefill only.
     */
    private boolean gpuActiveForDecode() {
        return useGpuForDecode && (gpuPipeline != null || gpuKernels != null);
    }

    private float[] decodeSingleToken(int tokenId) {
        int hidden = config.hiddenSize();
        int pos = cachedSeqLen;

        // Embedding lookup → decBuf
        if (tokenId >= 0 && tokenId < config.vocabSize()) {
            System.arraycopy(weights.embedTokens, tokenId * hidden, decBuf, 0, hidden);
        } else {
            Arrays.fill(decBuf, 0);
        }

        // Opt-C chained decode: ALL layers in ONE GPU submission
        if (useGpuForDecode && gpuPipeline != null && gpuPipeline.isAttnChainedEnabled()) {
            long t0 = System.nanoTime();
            gpuPipeline.decodeLayersChained(decBuf, decBuf, pos);
            profProjNs += System.nanoTime() - t0;

            // Final norm + lm head still on CPU (or GPU via separate submission)
            t0 = System.nanoTime();
            rmsNorm(decBuf, weights.finalNormWeight, config.rmsNormEps());
            profNormNs += System.nanoTime() - t0;

            t0 = System.nanoTime();
            if (gpuPipeline.hasLmHead()) {
                gpuPipeline.lmHead(decBuf, decLogits);
            } else {
                Arrays.fill(decLogits, 0);
                weights.lmHead.matvec(decBuf, decLogits);
            }
            profLmHeadNs += System.nanoTime() - t0;

            cachedSeqLen = pos + 1;
            return decLogits;
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
        if (useGpuForDecode && gpuPipeline != null && gpuPipeline.hasLmHead()) {
            gpuPipeline.lmHead(decBuf, decLogits);
        } else if (useGpuForDecode && gpuKernels != null && gpuKernels.hasLmHead()) {
            gpuKernels.lmHead().matvec(decBuf, decLogits);
        } else {
            Arrays.fill(decLogits, 0);
            weights.lmHead.matvec(decBuf, decLogits);
        }
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

        // Determine execution path for this layer.
        // HYBRID mode: GPU is reserved for prefill — single-token decode stays on CPU,
        // because on Intel iGPU each GPU dispatch costs 10–40 ms of fence-wait overhead
        // and 48 of them per token dominates the whole decode budget.
        final boolean usePipeline = useGpuForDecode
                && gpuPipeline != null && gpuPipeline.hasLayer(layerIdx);
        final boolean useKernels = !usePipeline && useGpuForDecode
                && gpuKernels != null && gpuKernels.hasLayer(layerIdx);
        // Opt-B: full GPU-resident decode (QKV+bias + RoPE + KV-append +
        // GQA attention + o_proj + MLP in ONE submission). Activates only when
        // -Dqwen.gpu.attention=true was set at startup AND all decoder layers
        // have GPU kernels AND the KV cache was successfully attached.
        final boolean useAttnResident = usePipeline
                && gpuPipeline.isAttnGpuResidentEnabled();

        // ── Pre-attention RMSNorm → decNormed ────────────────────────
        t0 = System.nanoTime();
        System.arraycopy(hiddenIo, 0, decNormed, 0, hidden);
        rmsNorm(decNormed, lw.inputNormWeight(), config.rmsNormEps());
        profNormNs += System.nanoTime() - t0;

        if (useAttnResident) {
            // Single-submission fast path: decNormed → QKV (+bias) → RoPE → GQA → MLP
            // → hiddenIo. CPU KV cache for this layer is intentionally NOT updated;
            // all subsequent decodes must keep going through this same fast path.
            t0 = System.nanoTime();
            gpuPipeline.decodeLayerGpuResident(layerIdx, decNormed, hiddenIo, hiddenIo, pos);
            profProjNs += System.nanoTime() - t0;
            return;
        }

        // ── Q/K/V Projections ────────────────────────────────────────
        t0 = System.nanoTime();
        if (usePipeline) {
            // V2.0: shared pipeline - 1 fence wait for QKV
            gpuPipeline.qkvFused(layerIdx, decNormed, decQKV);
            System.arraycopy(decQKV, 0, decQ, 0, qSize);
            System.arraycopy(decQKV, qSize, decK, 0, kvSize);
            System.arraycopy(decQKV, qSize + kvSize, decV, 0, kvSize);
        } else if (useKernels) {
            // V1: per-kernel matvec
            gpuKernels.qkvFused(layerIdx).matvec(decNormed, decQKV);
            System.arraycopy(decQKV, 0, decQ, 0, qSize);
            System.arraycopy(decQKV, qSize, decK, 0, kvSize);
            System.arraycopy(decQKV, qSize + kvSize, decV, 0, kvSize);
        } else {
            Arrays.fill(decQ, 0);
            Arrays.fill(decK, 0);
            Arrays.fill(decV, 0);
            lw.qProj().matvec(decNormed, decQ);
            lw.kProj().matvec(decNormed, decK);
            lw.vProj().matvec(decNormed, decV);
        }
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

        // ── Grouped-Query Attention (per-head parallel + SIMD) ────────────
        t0 = System.nanoTime();
        Arrays.fill(decAttnOut, 0, qSize, 0.0f);
        final float scale = (float) (1.0 / Math.sqrt(headDim));
        final int posLocal = pos;
        final int headDimLocal = headDim;
        final float[][] cacheK = kvCacheK[layerIdx];
        final float[][] cacheV = kvCacheV[layerIdx];

        IntStream.range(0, numHeads).parallel().forEach(h -> {
            int kvH = kvHeadForQueryHead(h);
            int qOff = h * headDimLocal;
            int outOff = h * headDimLocal;
            float[] kHead = cacheK[kvH];
            float[] vHead = cacheV[kvH];
            float[] scores = new float[posLocal + 1];
            for (int p = 0; p <= posLocal; p++) {
                scores[p] = SimdOps.dot(decQ, qOff, kHead, p * headDimLocal, headDimLocal) * scale;
            }
            softmax(scores, posLocal + 1);
            for (int p = 0; p <= posLocal; p++) {
                float w = scores[p];
                if (w < 1e-8f) continue;
                SimdOps.axpy(decAttnOut, outOff, w, vHead, p * headDimLocal, headDimLocal);
            }
        });
        profAttnNs += System.nanoTime() - t0;

        // ── O Proj + MLP block ────────────────────────────────────────────
        t0 = System.nanoTime();

        if (usePipeline && gpuPipeline.isMlpBatchEnabled()) {
            // ── V2.0 MLP batch: o_proj + residual1 + rmsNorm2 + gateUp + swiglu + down + residual2
            //    All 6 ops in ONE GPU submission → 1 fence wait instead of 3
            gpuPipeline.batchMlp(decAttnOut, hiddenIo, hiddenIo, layerIdx);
            profProjNs += System.nanoTime() - t0;
            return;  // Layer complete — hiddenIo holds the updated hidden state
        }

        // ── Fallback: individual dispatches (V1 or pipeline-without-batch) ─
        if (usePipeline) {
            gpuPipeline.oProj(layerIdx, decAttnOut, decOProj);
        } else if (useKernels) {
            Arrays.fill(decOProj, 0);
            gpuKernels.oProj(layerIdx).matvec(decAttnOut, decOProj);
        } else {
            Arrays.fill(decOProj, 0);
            lw.oProj().matvec(decAttnOut, decOProj);
        }
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
        if (usePipeline) {
            gpuPipeline.gateUpFused(layerIdx, decPostNorm, decGateUp);
        } else if (useKernels) {
            gpuKernels.gateUpFused(layerIdx).matvec(decPostNorm, decGateUp);
        } else {
            Arrays.fill(decGate, 0);
            Arrays.fill(decUp, 0);
            lw.gateProj().matvec(decPostNorm, decGate);
            lw.upProj().matvec(decPostNorm, decUp);
        }
        profProjNs += System.nanoTime() - t0;

        // SwiGLU activation: silu(gate) * up
        t0 = System.nanoTime();
        int intermediate = config.intermediateSize();
        boolean gpuFused = usePipeline || useKernels;
        for (int i = 0; i < intermediate; i++) {
            float gate = gpuFused ? decGateUp[i] : decGate[i];
            float up = gpuFused ? decGateUp[intermediate + i] : decUp[i];
            decMlpAct[i] = fastSilu(gate) * up;
        }
        profActNs += System.nanoTime() - t0;

        // down_proj
        t0 = System.nanoTime();
        if (usePipeline) {
            gpuPipeline.downProj(layerIdx, decMlpAct, decDown);
        } else if (useKernels) {
            Arrays.fill(decDown, 0);
            gpuKernels.downProj(layerIdx).matvec(decMlpAct, decDown);
        } else {
            Arrays.fill(decDown, 0);
            lw.downProj().matvec(decMlpAct, decDown);
        }
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
        if (gpuPipeline != null && gpuPipeline.hasLayer(layerIdx) && gpuPipeline.supportsBatch(layerIdx)) {
            // Opt-A: ONE batched GPU dispatch over the whole sequence (was seqLen dispatches).
            int stride = gpuKernels.qkvFusedN;
            float[] qkvBatch = new float[seqLen * stride];
            long tQkv = System.nanoTime();
            gpuPipeline.qkvFusedBatch(layerIdx, normed, qkvBatch, seqLen);
            profPrefillQkvBatchNs += System.nanoTime() - tQkv;
            for (int s = 0; s < seqLen; s++) {
                int base = s * stride;
                System.arraycopy(qkvBatch, base, q, s * qSize, qSize);
                System.arraycopy(qkvBatch, base + qSize, k, s * kvSize, kvSize);
                System.arraycopy(qkvBatch, base + qSize + kvSize, v, s * kvSize, kvSize);
            }
        } else if (gpuKernels != null && gpuKernels.hasLayer(layerIdx)) {
            // V1 / non-batched fallback: per-token GPU dispatch.
            float[] row = new float[hidden];
            float[] qkvRow = new float[gpuKernels.qkvFusedN];
            for (int s = 0; s < seqLen; s++) {
                System.arraycopy(normed, s * hidden, row, 0, hidden);
                if (gpuPipeline != null) {
                    gpuPipeline.qkvFused(layerIdx, row, qkvRow);
                } else {
                    gpuKernels.qkvFused(layerIdx).matvec(row, qkvRow);
                }
                System.arraycopy(qkvRow, 0, q, s * qSize, qSize);
                System.arraycopy(qkvRow, qSize, k, s * kvSize, kvSize);
                System.arraycopy(qkvRow, qSize + kvSize, v, s * kvSize, kvSize);
            }
        } else {
            lw.qProj().matmul(normed, q, seqLen);
            lw.kProj().matmul(normed, k, seqLen);
            lw.vProj().matmul(normed, v, seqLen);
        }
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

        // ── Causal Self-Attention (GQA, parallel over s×h + SIMD) ──────
        final float[] attnOut = new float[seqLen * qSize];
        final float scale = (float) (1.0 / Math.sqrt(headDim));
        final int qSizeLocal = qSize;
        final int headDimLocal = headDim;
        final int numHeadsLocal = numHeads;
        final int startPosLocal = startPos;
        final float[] qLocal = q;
        final float[][] cacheK = kvCacheK[layerIdx];
        final float[][] cacheV = kvCacheV[layerIdx];
        final int totalTasks = seqLen * numHeadsLocal;

        long tAttn = System.nanoTime();
        IntStream.range(0, totalTasks).parallel().forEach(idx -> {
            int s = idx / numHeadsLocal;
            int h = idx - s * numHeadsLocal;
            int queryPos = startPosLocal + s;
            int kvH = kvHeadForQueryHead(h);
            int qOff = s * qSizeLocal + h * headDimLocal;
            int outOff = qOff;
            float[] kHead = cacheK[kvH];
            float[] vHead = cacheV[kvH];
            float[] scores = new float[queryPos + 1];
            for (int p = 0; p <= queryPos; p++) {
                scores[p] = SimdOps.dot(qLocal, qOff, kHead, p * headDimLocal, headDimLocal) * scale;
            }
            softmax(scores, scores.length);
            for (int p = 0; p <= queryPos; p++) {
                float w = scores[p];
                if (w < 1e-8f) continue;
                SimdOps.axpy(attnOut, outOff, w, vHead, p * headDimLocal, headDimLocal);
            }
        });
        profPrefillAttnNs += System.nanoTime() - tAttn;

        // ── O projection ─────────────────────────────────────────────
        float[] oProjOut = new float[seqLen * hidden];
        if (gpuPipeline != null && gpuPipeline.hasLayer(layerIdx) && gpuPipeline.supportsBatch(layerIdx)) {
            // Opt-A: batched o_proj over the whole sequence (was seqLen dispatches).
            long tOProj = System.nanoTime();
            gpuPipeline.oProjBatch(layerIdx, attnOut, oProjOut, seqLen);
            profPrefillOProjBatchNs += System.nanoTime() - tOProj;
        } else if (gpuKernels != null && gpuKernels.hasLayer(layerIdx)) {
            float[] row = new float[qSize];
            float[] tmpOut = new float[hidden];
            for (int s = 0; s < seqLen; s++) {
                System.arraycopy(attnOut, s * qSize, row, 0, qSize);
                if (gpuPipeline != null) {
                    gpuPipeline.oProj(layerIdx, row, tmpOut);
                } else {
                    gpuKernels.oProj(layerIdx).matvec(row, tmpOut);
                }
                System.arraycopy(tmpOut, 0, oProjOut, s * hidden, hidden);
            }
        } else {
            lw.oProj().matmul(attnOut, oProjOut, seqLen);
        }

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
        if (gpuPipeline != null && gpuPipeline.hasLayer(layerIdx) && gpuPipeline.supportsBatch(layerIdx)) {
            // Opt-A: batched gate+up over the whole sequence (was seqLen dispatches).
            int stride = gpuKernels.gateUpFusedN; // 2 * intermediate
            float[] guBatch = new float[seqLen * stride];
            long tGu = System.nanoTime();
            gpuPipeline.gateUpFusedBatch(layerIdx, postNormed, guBatch, seqLen);
            profPrefillGuBatchNs += System.nanoTime() - tGu;
            for (int s = 0; s < seqLen; s++) {
                int base = s * stride;
                System.arraycopy(guBatch, base, gate, s * intermediate, intermediate);
                System.arraycopy(guBatch, base + intermediate, up, s * intermediate, intermediate);
            }
        } else if (gpuKernels != null && gpuKernels.hasLayer(layerIdx)) {
            float[] row = new float[hidden];
            float[] guRow = new float[gpuKernels.gateUpFusedN]; // [2 * intermediate]
            for (int s = 0; s < seqLen; s++) {
                System.arraycopy(postNormed, s * hidden, row, 0, hidden);
                if (gpuPipeline != null) {
                    gpuPipeline.gateUpFused(layerIdx, row, guRow);
                } else {
                    gpuKernels.gateUpFused(layerIdx).matvec(row, guRow);
                }
                System.arraycopy(guRow, 0, gate, s * intermediate, intermediate);
                System.arraycopy(guRow, intermediate, up, s * intermediate, intermediate);
            }
        } else {
            lw.gateProj().matmul(postNormed, gate, seqLen);
            lw.upProj().matmul(postNormed, up, seqLen);
        }

        // SwiGLU: silu(gate) * up
        float[] mlpActivation = new float[seqLen * intermediate];
        for (int s = 0; s < seqLen; s++) {
            int off = s * intermediate;
            for (int i = 0; i < intermediate; i++) {
                float g = gate[off + i];
                mlpActivation[off + i] = fastSilu(g) * up[off + i];
            }
        }

        // down_proj
        float[] downOut = new float[seqLen * hidden];
        if (gpuPipeline != null && gpuPipeline.hasLayer(layerIdx) && gpuPipeline.supportsBatch(layerIdx)) {
            // Opt-A: batched down_proj over the whole sequence (was seqLen dispatches).
            long tDown = System.nanoTime();
            gpuPipeline.downProjBatch(layerIdx, mlpActivation, downOut, seqLen);
            profPrefillDownBatchNs += System.nanoTime() - tDown;
        } else if (gpuKernels != null && gpuKernels.hasLayer(layerIdx)) {
            float[] row = new float[intermediate];
            float[] tmpOut = new float[hidden];
            for (int s = 0; s < seqLen; s++) {
                System.arraycopy(mlpActivation, s * intermediate, row, 0, intermediate);
                if (gpuPipeline != null) {
                    gpuPipeline.downProj(layerIdx, row, tmpOut);
                } else {
                    gpuKernels.downProj(layerIdx).matvec(row, tmpOut);
                }
                System.arraycopy(tmpOut, 0, downOut, s * hidden, hidden);
            }
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

    /**
     * Fast SiLU (Swish) activation: {@code x * sigmoid(x) = x / (1 + exp(-x))}.
     *
     * <p>Uses the Schraudolph (1999) bit-trick for {@code exp}, which is ~3–5×
     * faster than {@link Math#exp} at the cost of ≤3 % relative error in
     * {@code sigmoid}. This error is well below INT4 quantisation noise and
     * has no measurable impact on greedy-decoding output quality.
     *
     * <p>Clamps to exact values for |x| ≥ 10 (sigmoid ≈ 0 or 1 there).
     */
    static float fastSilu(float x) {
        if (x >= 10.0f) return x;      // sigmoid(10) > 0.9999 → silu ≈ x
        if (x <= -10.0f) return 0.0f;   // sigmoid(-10) < 0.0001 → silu ≈ 0
        // exp(-x) via Schraudolph bit-trick:  exp(t) ≈ Float.intBitsToFloat((int)(t·2²³/ln2 + 127·2²³))
        int bits = (int) (-x * 12102203.161561485f + 1065353216.0f);
        float expNeg = Float.intBitsToFloat(bits < 0 ? 0 : bits);
        return x / (1.0f + expNeg);
    }

    static void rmsNorm(float[] x, float[] weight, float eps) {
        float sumSq = 0;
        for (float v : x) sumSq += v * v;
        float rms = (float) (1.0 / Math.sqrt(sumSq / x.length + eps));
        for (int i = 0; i < x.length; i++) {
            x[i] = x[i] * rms * weight[i];
        }
    }

    /**
     * Add bias vector to a single-position output: out[i] += bias[i].
     */
    private static void addBias(float[] out, float[] bias) {
        for (int i = 0; i < bias.length; i++) {
            out[i] += bias[i];
        }
    }

    /**
     * Add bias vector to batched output: out[s * dim + i] += bias[i] for each sequence position.
     */
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

    /**
     * Apply a repetition penalty to all previously-generated token logits.
     * Uses a small {@link HashSet} (typically &lt;= maxTokens, i.e. a few hundred
     * entries) to deduplicate — the previous implementation allocated a fresh
     * {@code boolean[vocabSize]} (= 594 KB for Qwen 0.5B's 151 936-word vocab)
     * per token and burned ~120 MB of GC pressure over a 200-token decode.
     */
    private void applyRepetitionPenalty(float[] logits, int[] generatedIds, int count) {
        Set<Integer> seen = new HashSet<>(count * 2);
        for (int i = 0; i < count; i++) {
            int id = generatedIds[i];
            if (id < 0 || id >= logits.length) continue;
            if (!seen.add(id)) continue;
            float v = logits[id];
            logits[id] = v > 0 ? v / repetitionPenalty : v * repetitionPenalty;
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
        int gpuL = gpuPipeline != null ? (gpuPipeline.hasLayer(config.numHiddenLayers() - 1)
                ? config.numHiddenLayers() : gpuKernels != null ? gpuKernels.getGpuLayers() : 0)
                : gpuKernels != null ? gpuKernels.getGpuLayers() : 0;
        boolean pipelineV2 = gpuPipeline != null && gpuPipeline.isMlpBatchEnabled();

        return String.format(
                "[Qwen2 Decode Profile] %d tokens, %.1f ms total, %.1f ms/token%n"
                        + "  Mode:          %s%n"
                        + "  Prefill:       %.1f ms%n"
                        + "  Projections:   %.1f ms avg (%.0f%%)%n"
                        + "  Attention:     %.1f ms avg (%.0f%%)%n"
                        + "  Norms+RoPE:    %.1f ms avg (%.0f%%)%n"
                        + "  SwiGLU:        %.1f ms avg (%.0f%%)%n"
                        + "  LM head:       %.1f ms avg (%.0f%%)",
                profSteps, totalMs, perToken,
                gpuL > 0
                        ? (pipelineV2
                        ? "GPU-V2 (pipeline, " + gpuL + " layers, mlpBatch=true, 48 submits/token)"
                        : "GPU-V1 (" + gpuL + " layers, lmHead=" + (gpuKernels != null && gpuKernels.hasLmHead()) + ")")
                        : "CPU-only",
                profPrefillNs / 1e6,
                profProjNs / 1e6 / profSteps, 100.0 * profProjNs / pctDivisor,
                profAttnNs / 1e6 / profSteps, 100.0 * profAttnNs / pctDivisor,
                profNormNs / 1e6 / profSteps, 100.0 * profNormNs / pctDivisor,
                profActNs / 1e6 / profSteps, 100.0 * profActNs / pctDivisor,
                profLmHeadNs / 1e6 / profSteps, 100.0 * profLmHeadNs / pctDivisor
        );
    }
}
