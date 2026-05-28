package com.aresstack.windirectml.inference.qwen;

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
 *   <li>Formats prompts using {@link QwenChatTemplate} (ChatML)</li>
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
 * </ul>
 *
 * <h2>System properties</h2>
 * <ul>
 *   <li>{@code qwen.gpu.layers} — number of decoder layers on GPU (default: all)</li>
 *   <li>{@code qwen.gpu.lmhead} — place lm_head on GPU (default: {@code false}
 *       to save ~544 MB VRAM on small iGPUs)</li>
 * </ul>
 */
public class QwenInferenceEngine implements InferenceEngine {

    private static final Logger log = LoggerFactory.getLogger(QwenInferenceEngine.class);

    private final Path modelDir;
    private final int defaultMaxTokens;
    private final String backend;   // "directml" | "cpu" | "auto"

    private Qwen2Config config;
    private QwenTokenizer tokenizer;
    private Qwen2Weights weights;
    private Qwen2Runtime runtime;
    private boolean ready = false;

    // GPU resources (initialised during initialize() when backend != "cpu")
    private WindowsBindings wb;
    private QwenGpuKernels gpuKernels;
    private QwenGpuPipeline gpuPipeline;  // V2.0 shared pipeline (batched MLP)

    /**
     * Create a new Qwen inference engine (CPU-only).
     *
     * @param modelDir        path to the model directory
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
        this.modelDir = modelDir;
        this.defaultMaxTokens = defaultMaxTokens > 0 ? defaultMaxTokens : 256;
        this.backend = backend != null ? backend : "cpu";
    }

    @Override
    public void initialize() throws InferenceException {
        log.info("QwenInferenceEngine initializing from {}", modelDir);

        // Validate model directory
        String missing = QwenModelDirValidator.describeMissingModelFile(modelDir);
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
            weights = Qwen2Weights.load(modelDir, config);

            // ── GPU acceleration ──────────────────────────────────────
            if (!"cpu".equalsIgnoreCase(backend) && WindowsBindings.isSupported()) {
                try {
                    wb = new WindowsBindings();
                    wb.init(backend);
                    if (wb.hasDirectMl()) {
                        int gpuLayerCount = Integer.getInteger("qwen.gpu.layers",
                                config.numHiddenLayers());
                        boolean gpuLmHead = Boolean.parseBoolean(
                                System.getProperty("qwen.gpu.lmhead", "false"));
                        gpuKernels = QwenGpuKernels.create(
                                wb, weights, config, gpuLayerCount, gpuLmHead);
                        // V2.0: Create shared GPU pipeline (batched MLP, 48 fence-waits/token)
                        try {
                            gpuPipeline = new QwenGpuPipeline(wb, gpuKernels, config);
                            gpuPipeline.uploadLayerWeights(wb, weights, config);
                            log.info("GPU acceleration: {}/{} layers on GPU, lmHead={}, pipeline=V2.0 (mlpBatch={})",
                                    gpuKernels.getGpuLayers(), config.numHiddenLayers(),
                                    gpuKernels.hasLmHead(), gpuPipeline.isMlpBatchEnabled());
                        } catch (Exception pe) {
                            log.warn("GPU pipeline V2.0 failed, using V1 per-kernel dispatch: {}",
                                    pe.getMessage());
                            gpuPipeline = null;
                            log.info("GPU acceleration: {}/{} layers on GPU (V1), lmHead={}",
                                    gpuKernels.getGpuLayers(), config.numHiddenLayers(),
                                    gpuKernels.hasLmHead());
                        }
                    } else {
                        log.warn("DirectML device not available, falling back to CPU");
                        cleanupGpu();
                    }
                } catch (Exception e) {
                    if ("auto".equalsIgnoreCase(backend)) {
                        log.warn("GPU init failed, falling back to CPU: {}", e.getMessage());
                        cleanupGpu();
                    } else {
                        cleanupGpu();
                        throw new InferenceException(
                                "GPU initialization failed: " + e.getMessage(), e);
                    }
                }
            }

            // Create runtime (gpuPipeline + gpuKernels may be null → CPU-only)
            runtime = new Qwen2Runtime(config, weights, tokenizer, gpuKernels, gpuPipeline);

            long elapsed = System.currentTimeMillis() - t0;
            log.info("QwenInferenceEngine initialized in {} ms (backend={})",
                    elapsed, gpuPipeline != null
                            ? "GPU-V2(" + gpuKernels.getGpuLayers() + " layers, mlpBatch=" + gpuPipeline.isMlpBatchEnabled() + ")"
                            : gpuKernels != null ? "GPU-V1(" + gpuKernels.getGpuLayers() + " layers)" : "CPU");
            ready = true;

        } catch (Exception e) {
            cleanupFailedInitialization();
            throw new InferenceException("Failed to initialize Qwen engine: " + e.getMessage(), e);
        }
    }

    @Override
    public InferenceResult generate(InferenceRequest request) throws InferenceException {
        if (!ready) {
            throw new InferenceException("Qwen engine not initialized");
        }

        try {
            // Format prompt using ChatML template
            String systemPrompt = request.getSystemPrompt();
            String userMessage = request.getUserPrompt();

            String formattedPrompt = QwenChatTemplate.formatChat(
                    systemPrompt != null && !systemPrompt.isBlank() ? systemPrompt : null,
                    userMessage
            );

            int maxTokens = request.getMaxTokens() > 0 ? request.getMaxTokens() : defaultMaxTokens;
            log.info("Generating: maxTokens={}, promptLen={} chars", maxTokens, formattedPrompt.length());

            long t0 = System.currentTimeMillis();
            String generatedText = runtime.generate(formattedPrompt, maxTokens);
            long elapsed = System.currentTimeMillis() - t0;

            // Count tokens for usage. This is approximate because generated text
            // can re-tokenize differently from the emitted token IDs.
            int promptTokens = tokenizer.encode(formattedPrompt).length;
            int completionTokens = tokenizer.encode(generatedText).length;

            log.info("Generated {} chars ({} tokens approx) in {} ms ({} ms/token approx)",
                    generatedText.length(), completionTokens, elapsed,
                    completionTokens > 0 ? elapsed / completionTokens : 0);

            String finishReason = completionTokens >= maxTokens ? "max_tokens" : "end_turn";

            return new InferenceResult(
                    generatedText.strip(),
                    finishReason,
                    new InferenceResult.Usage(promptTokens, completionTokens,
                            promptTokens + completionTokens)
            );

        } catch (Exception e) {
            throw new InferenceException("Qwen generation failed: " + e.getMessage(), e);
        }
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

    /** Returns the underlying runtime (for testing/profiling). */
    public Qwen2Runtime getRuntime() {
        return runtime;
    }

    /** Returns the loaded config (for testing/diagnostics). */
    public Qwen2Config getConfig() {
        return config;
    }
}
