package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.GeneratedToken;
import com.aresstack.windirectml.inference.GenerationTokenSink;
import com.aresstack.windirectml.inference.InferenceEngine;
import com.aresstack.windirectml.inference.InferenceException;
import com.aresstack.windirectml.inference.InferenceRequest;
import com.aresstack.windirectml.inference.InferenceResult;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;

/**
 * Inference engine for Qwen2.5-Coder-Instruct models (CPU-only).
 *
 * <p>Implements the {@link InferenceEngine} interface for Qwen2.5-Coder
 * text generation. This engine:
 * <ul>
 *   <li>Loads config, tokenizer, and weights from a model directory</li>
 *   <li>Formats prompts through the shared prompt pipeline ({@code PromptStrategies})</li>
 *   <li>Generates text using {@link Qwen2Runtime} (greedy CPU decode)</li>
 *   <li>Stops on {@link QwenStopTokenPolicy} tokens</li>
 * </ul>
 *
 * <h2>Model directory layout (per issue #100 contract)</h2>
 * <pre>
 * model/qwen2.5-coder-0.5b-directml-int4/
 *   config.json              — HuggingFace model config
 *   tokenizer.json           — HuggingFace BPE tokenizer
 *   tokenizer_config.json    — Tokenizer config with ChatML template
 *   special_tokens_map.json  — Special token definitions
 *   model.onnx               — ONNX model graph (weight metadata)
 *   model.onnx.data          — External weight data (memory-mapped)
 * </pre>
 *
 * <h2>Status</h2>
 * <p>This engine is <b>CPU-only</b> and part of the Qwen runtime bring-up.
 * The DirectML Workbench wires it as a manual test path so local release-jar
 * builds can exercise Qwen generation. The model remains planned/not-shipped
 * in the global registry until the ONNX source/layout and end-to-end generation
 * are verified with real weights.</p>
 *
 * <h2>Differences from Phi-3 engine</h2>
 * <ul>
 *   <li>ChatML template instead of Phi-3 role tokens</li>
 *   <li>Qwen BPE tokenizer (byte-level, ~152k vocab) instead of SentencePiece</li>
 *   <li>DirectML GPU path via {@link QwenGpuKernels} (fused QKV + fused gate+up)</li>
 *   <li>GQA attention with separate Q/KV head counts (attention stays on CPU)</li>
 * </ul>
 *
 * <h2>Backend selection</h2>
 * <ul>
 *   <li>{@code "directml"} — always use GPU; fail if DirectML is not available</li>
 *   <li>{@code "auto"} — try GPU, silently fall back to CPU</li>
 *   <li>{@code "cpu"} — always CPU; no GPU init</li>
 *   <li>{@code "hybrid"} — max GPU offload: same as AUTO but additionally
 *       forces lm_head onto the GPU (overrides {@code qwen.gpu.lmhead}).
 *       Use this on hardened/locked-down Windows hosts where CPU INT4
 *       SIMD is throttled by AV / AppLocker / thermal limits and even a
 *       slow Intel iGPU outperforms the CPU by 100x+. The historic
 *       "GPU prefill + CPU decode" semantic has been removed — it was
 *       counter-productive on exactly the hosts the mode was meant for.</li>
 * </ul>
 *
 * <h2>System properties</h2>
 * <ul>
 *   <li>{@code qwen.gpu.layers} — number of decoder layers on GPU (default: all)</li>
 *   <li>{@code qwen.gpu.lmhead} — place lm_head on GPU (default: {@code true};
 *       set to {@code false} only if VRAM is tight — the lm_head matrix is
 *       ~70 MB INT4 for Qwen 0.5B's 151 936-word vocab. On Intel iGPU hosts
 *       CPU lm_head has been measured at 40 s/token, which alone exceeds
 *       the sum of ALL other per-token work — GPU lm_head is essentially
 *       mandatory there.)</li>
 * </ul>
 */
public class QwenInferenceEngine implements InferenceEngine {

    private static final Logger log = LoggerFactory.getLogger(QwenInferenceEngine.class);

    private final Path modelDir;
    private final int defaultMaxTokens;
    private final String backend;   // "directml" | "cpu" | "auto" | "hybrid"
    private final String modelFileName;

    private Qwen2Config config;
    private QwenTokenizer tokenizer;
    private Qwen2Weights weights;
    private Qwen2Runtime runtime;
    private boolean ready = false;

    // GPU resources (initialised during initialize() when backend != "cpu")
    private WindowsBindings wb;
    private QwenGpuKernels gpuKernels;
    private QwenGpuPipeline gpuPipeline;  // V2.0 shared pipeline (batched MLP)
    private QwenGpuKvCache gpuKvCache;    // Opt-B GPU-resident KV cache (null unless qwen.gpu.attention=true)

    /**
     * Create a new Qwen inference engine (CPU-only).
     *
     * @param modelDir         path to the model directory
     * @param defaultMaxTokens default maximum tokens if not specified in request
     */
    public QwenInferenceEngine(Path modelDir, int defaultMaxTokens) {
        this(modelDir, defaultMaxTokens, "cpu");
    }

    /**
     * Create a new Qwen inference engine with backend selection.
     *
     * @param modelDir         path to the model directory
     * @param defaultMaxTokens default maximum tokens if not specified in request
     * @param backend          "directml", "auto", or "cpu"
     */
    public QwenInferenceEngine(Path modelDir, int defaultMaxTokens, String backend) {
        this(modelDir, defaultMaxTokens, backend, QwenModelDirValidator.DEFAULT_MODEL_FILE);
    }

    /**
     * Create a new Qwen inference engine with an explicit ONNX filename.
     *
     * @param modelDir         path to the model directory
     * @param defaultMaxTokens default maximum tokens if not specified in request
     * @param backend          "directml", "auto", "hybrid", or "cpu"
     * @param modelFileName    ONNX filename inside modelDir
     */
    public QwenInferenceEngine(Path modelDir, int defaultMaxTokens, String backend, String modelFileName) {
        this.modelDir = modelDir;
        this.defaultMaxTokens = defaultMaxTokens > 0 ? defaultMaxTokens : 256;
        this.backend = backend != null ? backend : "cpu";
        this.modelFileName = QwenModelDirValidator.normalizeModelFileName(modelFileName);
    }

    @Override
    public void initialize() throws InferenceException {
        log.info("QwenInferenceEngine initializing from {} using {}", modelDir, modelFileName);

        // Validate model directory
        String missing = QwenModelDirValidator.describeMissingRequiredFiles(modelDir, modelFileName);
        if (missing != null) {
            throw new InferenceException("Cannot initialize Qwen engine: " + missing);
        }

        try {
            long t0 = System.currentTimeMillis();

            // Load config
            Path configPath = modelDir.resolve("config.json");
            config = Qwen2Config.load(configPath);
            log.info("Config: hidden={}, layers={}, heads={} (kv={}), vocab={}, headDim={}",
                    config.hiddenSize(), config.numHiddenLayers(),
                    config.numAttentionHeads(), config.numKeyValueHeads(),
                    config.vocabSize(), config.headDim());

            // Load tokenizer
            Path tokenizerPath = modelDir.resolve("tokenizer.json");
            tokenizer = QwenTokenizer.load(tokenizerPath);
            log.info("Tokenizer loaded: vocabSize={}", tokenizer.vocabSize());

            // Load weights
            weights = Qwen2Weights.load(modelDir, config, modelFileName);

            // ── GPU acceleration ──────────────────────────────────────
            if (!"cpu".equalsIgnoreCase(backend) && WindowsBindings.isSupported()) {
                try {
                    wb = new WindowsBindings();
                    // "hybrid" is not a known native backend name — map to "auto"
                    // for the native init (the GPU is still required for prefill).
                    String nativeBackend = "hybrid".equalsIgnoreCase(backend) ? "auto" : backend;
                    wb.init(nativeBackend);
                    if (wb.hasDirectMl()) {
                        int gpuLayerCount = Integer.getInteger("qwen.gpu.layers",
                                config.numHiddenLayers());
                        // HYBRID forces lm_head onto the GPU regardless of the
                        // qwen.gpu.lmhead system property: on hardened/locked-down
                        // hosts CPU INT4 matvec of the 151 936 x 896 lm_head can
                        // dwarf ALL projection cost combined (we have measured
                        // 40 s/token on such a host). The ~70 MB INT4 VRAM cost
                        // is negligible on any GPU capable of running Qwen 0.5B.
                        // Default lmhead policy outside HYBRID is now also `true`
                        // for the same reason — the historic `false` default
                        // optimised for VRAM scarcity that no realistic Qwen-0.5B
                        // host actually faces.
                        boolean hybridMode = "hybrid".equalsIgnoreCase(backend);
                        boolean gpuLmHead = hybridMode || Boolean.parseBoolean(
                                System.getProperty("qwen.gpu.lmhead", "true"));
                        gpuKernels = QwenGpuKernels.create(
                                wb, weights, config, gpuLayerCount, gpuLmHead);
                        // V2.0: Create shared GPU pipeline (batched MLP, 48 fence-waits/token)
                        try {
                            gpuPipeline = new QwenGpuPipeline(wb, gpuKernels, config);
                            gpuPipeline.uploadLayerWeights(wb, weights, config);

                            // Opt-B: GPU-resident attention (KV cache + rope + GQA on GPU).
                            // ENABLED BY DEFAULT (2026-05-29) - drops decode from ~3.7 s/token to ~1 s/token
                            // on Intel iGPU. Disable with -Dqwen.gpu.attention=false if it causes issues.
                            //
                            // The KV cache is requested here but actually allocated LAZILY at the end
                            // of the first prefill, so it doesn't compete with matmulBatch scratch
                            // buffers for GPU memory (which would OOM Intel iGPUs).
                            boolean attnFlag = Boolean.parseBoolean(
                                    System.getProperty("qwen.gpu.attention", "true"));
                            if (attnFlag
                                    && gpuKernels.getGpuLayers() == config.numHiddenLayers()) {
                                int kvMaxSeq = Integer.getInteger(
                                        "qwen.gpu.attention.maxseqlen", 2048);
                                gpuPipeline.requestLazyGpuAttention(wb,
                                        config.numHiddenLayers(),
                                        config.numKeyValueHeads(),
                                        config.headDim(),
                                        kvMaxSeq);
                            } else if (attnFlag) {
                                log.warn("Opt-B requested but only {}/{} layers on GPU - GPU-resident attention disabled (set qwen.gpu.layers={})",
                                        gpuKernels.getGpuLayers(), config.numHiddenLayers(),
                                        config.numHiddenLayers());
                            }

                            log.info("GPU acceleration: {}/{} layers on GPU, lmHead={}, pipeline=V2.0 (mlpBatch={}, attnLazy={})",
                                    gpuKernels.getGpuLayers(), config.numHiddenLayers(),
                                    gpuKernels.hasLmHead(), gpuPipeline.isMlpBatchEnabled(),
                                    attnFlag);
                            releaseHostWeightsIfFullyOffloaded();
                        } catch (Exception pe) {
                            log.warn("GPU pipeline V2.0 failed, using V1 per-kernel dispatch: {}",
                                    pe.getMessage());
                            gpuPipeline = null;
                            log.info("GPU acceleration: {}/{} layers on GPU (V1), lmHead={}",
                                    gpuKernels.getGpuLayers(), config.numHiddenLayers(),
                                    gpuKernels.hasLmHead());
                            releaseHostWeightsIfFullyOffloaded();
                        }
                    } else {
                        log.warn("DirectML device not available, falling back to CPU");
                        cleanupGpu();
                    }
                } catch (Exception e) {
                    if ("auto".equalsIgnoreCase(backend) || "hybrid".equalsIgnoreCase(backend)) {
                        log.warn("GPU init failed, falling back to CPU: {}", e.getMessage());
                        cleanupGpu();
                    } else {
                        cleanupGpu();
                        throw new InferenceException(
                                "GPU initialization failed: " + e.getMessage(), e);
                    }
                }
            }

            // Create runtime. HYBRID semantics REVISED 2026-05:
            //   old: GPU prefill + CPU decode — turned out useless on hardened
            //        hosts where CPU INT4 matvec is 100-600x slower than GPU
            //        even with iGPU per-submission overhead.
            //   new: "max GPU offload" — same kernel use as AUTO/DIRECTML for
            //        both prefill and decode, PLUS lm_head on GPU. This is the
            //        only viable mode on hardened laptops; on free-running
            //        multi-core CPUs AUTO behaves identically now (lm_head
            //        default is also true). useGpuForDecode therefore stays true.
            final boolean hybridMode = "hybrid".equalsIgnoreCase(backend);
            runtime = new Qwen2Runtime(config, weights, tokenizer, gpuKernels, gpuPipeline,
                    /* useGpuForDecode */ true);

            long elapsed = System.currentTimeMillis() - t0;
            String backendDesc;
            if ("hybrid".equalsIgnoreCase(backend) && gpuPipeline != null) {
                backendDesc = "HYBRID(max GPU offload, "
                        + gpuKernels.getGpuLayers() + " layers + lm_head on GPU)";
            } else if (gpuPipeline != null) {
                backendDesc = "GPU-V2(" + gpuKernels.getGpuLayers() + " layers, mlpBatch="
                        + gpuPipeline.isMlpBatchEnabled() + ", lmHead=" + gpuKernels.hasLmHead() + ")";
            } else if (gpuKernels != null) {
                backendDesc = "GPU-V1(" + gpuKernels.getGpuLayers() + " layers, lmHead="
                        + gpuKernels.hasLmHead() + ")";
            } else {
                backendDesc = "CPU";
            }
            log.info("QwenInferenceEngine initialized in {} ms (backend={})", elapsed, backendDesc);
            ready = true;

        } catch (Exception e) {
            cleanupFailedInitialization();
            throw new InferenceException("Failed to initialize Qwen engine: " + e.getMessage(), e);
        }
    }

    @Override
    public InferenceResult generate(InferenceRequest request) throws InferenceException {
        return generate(request, null);
    }

    @Override
    public InferenceResult generate(InferenceRequest request, GenerationTokenSink sink) throws InferenceException {
        if (!ready) {
            throw new InferenceException("Qwen engine not initialized");
        }

        try {
            String formattedPrompt = formatPrompt(request);
            int maxTokens = request.getMaxTokens() > 0 ? request.getMaxTokens() : defaultMaxTokens;
            log.info("Generating: maxTokens={}, promptLen={} chars", maxTokens, formattedPrompt.length());

            long t0 = System.currentTimeMillis();
            final int[] streamedTokens = {0};
            Qwen2Runtime.TokenConsumer consumer = sink == null ? null : (tokenId, fullText, delta) -> {
                streamedTokens[0]++;
                sink.onToken(new GeneratedToken(tokenId, fullText, delta));
            };
            String generatedText = runtime.generateStreaming(formattedPrompt, maxTokens, consumer);
            long elapsed = System.currentTimeMillis() - t0;

            int promptTokens = tokenizer.encode(formattedPrompt).length;
            int completionTokens = streamedTokens[0] > 0
                    ? streamedTokens[0]
                    : tokenizer.encode(generatedText).length;

            log.info("Generated {} chars ({} tokens approx) in {} ms ({} ms/token approx)",
                    generatedText.length(), completionTokens, elapsed,
                    completionTokens > 0 ? elapsed / completionTokens : 0);

            String finishReason = completionTokens >= maxTokens ? "max_tokens" : "end_turn";
            InferenceResult result = new InferenceResult(
                    generatedText.strip(),
                    finishReason,
                    new InferenceResult.Usage(promptTokens, completionTokens,
                            promptTokens + completionTokens)
            );
            if (sink != null) {
                sink.onCompleted(result);
            }
            return result;

        } catch (Exception e) {
            throw new InferenceException("Qwen generation failed: " + e.getMessage(), e);
        }
    }

    private String formatPrompt(InferenceRequest request) {
        return com.aresstack.windirectml.inference.prompt.PromptStrategies
                .forModel(request.getModelId())
                .renderPrompt(new com.aresstack.windirectml.inference.prompt.PromptInput(
                        request.getTask(), request.getUserPrompt(), request.getSystemPrompt()));
    }


    @Override
    public void shutdown() {
        log.info("Shutting down QwenInferenceEngine");
        ready = false;
        cleanupGpu();
        if (weights != null) {
            try {
                weights.close();
            } catch (Exception e) {
                log.warn("Error closing weights: {}", e.getMessage());
            }
        }
        runtime = null;
        weights = null;
        tokenizer = null;
        config = null;
    }

    private void releaseHostWeightsIfFullyOffloaded() {
        boolean enabled = Boolean.parseBoolean(
                System.getProperty("qwen.heapLight.releaseUploadedWeights", "true"));
        if (!enabled || weights == null || gpuKernels == null) {
            return;
        }
        if (gpuKernels.getGpuLayers() != config.numHiddenLayers()) {
            log.info("Heap-light host weight release skipped: only {}/{} layers are on GPU",
                    gpuKernels.getGpuLayers(), config.numHiddenLayers());
            return;
        }
        long released = weights.releaseGpuUploadedProjectionStorage(gpuKernels.hasLmHead());
        if (released > 0) {
            log.info("Heap-light mode active: Java keeps embeddings/norms/biases only; uploaded projections are GPU-resident handles");
        }
    }

    /**
     * Clean up GPU resources (idempotent).
     */
    private void cleanupGpu() {
        if (gpuPipeline != null) {
            try {
                gpuPipeline.close();
            } catch (Exception e) {
                log.warn("Error closing GPU pipeline: {}", e.getMessage());
            }
            gpuPipeline = null;
        }
        // KV cache lifetime is now owned by the pipeline (lazy allocation).
        // The legacy gpuKvCache field is kept null and ignored.
        gpuKvCache = null;
        if (gpuKernels != null) {
            try {
                gpuKernels.close();
            } catch (Exception e) {
                log.warn("Error closing GPU kernels: {}", e.getMessage());
            }
            gpuKernels = null;
        }
        if (wb != null) {
            try {
                wb.close();
            } catch (Exception e) {
                log.warn("Error closing WindowsBindings: {}", e.getMessage());
            }
            wb = null;
        }
    }

    private void cleanupFailedInitialization() {
        ready = false;
        cleanupGpu();
        if (weights != null) {
            try {
                weights.close();
            } catch (Exception closeError) {
                log.warn("Error closing weights after failed initialization: {}", closeError.getMessage());
            }
        }
        runtime = null;
        weights = null;
        tokenizer = null;
        config = null;
    }

    @Override
    public boolean isReady() {
        return ready;
    }

    /**
     * Returns the underlying runtime (for testing/profiling).
     */
    public Qwen2Runtime getRuntime() {
        return runtime;
    }

    /**
     * Returns the loaded config (for testing/diagnostics).
     */
    public Qwen2Config getConfig() {
        return config;
    }
}
