package com.aresstack.windirectml.inference;

import com.aresstack.windirectml.config.InferenceConfiguration;
import com.aresstack.windirectml.inference.phi3.Phi3Config;
import com.aresstack.windirectml.inference.phi3.Phi3GpuKernels;
import com.aresstack.windirectml.inference.phi3.Phi3GpuPipeline;
import com.aresstack.windirectml.inference.phi3.Phi3Runtime;
import com.aresstack.windirectml.inference.phi3.Phi3Tokenizer;
import com.aresstack.windirectml.inference.phi3.Phi3Weights;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * {@link InferenceEngine} implementation for Phi-3-mini-4k-instruct.
 * <p>
 * Loads the Phi-3 model from the DirectML INT4 AWQ variant and runs
 * text generation using the decoder runtime ({@link Phi3Runtime}).
 * <p>
 * <b>Execution modes:</b>
 * <ul>
 *   <li><b>CPU</b> ({@code backend="cpu"}): all matrix multiplications on CPU</li>
 *   <li><b>DirectML</b> ({@code backend="directml"}): projections dispatched to GPU
 *       via {@link Phi3GpuKernels}; attention/norms/activations on CPU</li>
 *   <li><b>Auto</b> ({@code backend="auto"}): try DirectML, fall back to CPU</li>
 * </ul>
 * <p>
 * <b>Model requirements:</b>
 * <ul>
 *   <li>{@code config.json} — HuggingFace model config</li>
 *   <li>{@code tokenizer.json} — HuggingFace tokenizer</li>
 *   <li>{@code model.onnx} — ONNX model proto (weight metadata)</li>
 *   <li>{@code model.onnx.data} — External weight data (~2.1 GB)</li>
 * </ul>
 * <p>
 * <b>System properties:</b>
 * <ul>
 *   <li>{@code phi3.gpu.layers} — number of decoder layers to place on GPU
 *       (default: all layers)</li>
 *   <li>{@code phi3.gpu.lmhead} — put lm_head projection on GPU
 *       (default: {@code true})</li>
 * </ul>
 */
public class Phi3InferenceEngine implements InferenceEngine {

    private static final Logger log = LoggerFactory.getLogger(Phi3InferenceEngine.class);

    private final Path modelDir;
    private final int defaultMaxTokens;
    private final String backend;   // "directml" | "cpu" | "auto"

    private Phi3Config config;
    private Phi3Tokenizer tokenizer;
    private Phi3Weights weights;
    private Phi3Runtime runtime;

    // GPU resources (created during initialize if backend != "cpu")
    private WindowsBindings wb;
    private Phi3GpuKernels gpuKernels;
    private Phi3GpuPipeline gpuPipeline;  // V2.0 shared pipeline

    private boolean ready = false;

    /**
     * CPU-only convenience constructor.
     *
     * @param modelDir path to the directory containing model files
     */
    public Phi3InferenceEngine(Path modelDir) {
        this(modelDir, 256, "cpu");
    }

    /**
     * CPU-only convenience constructor with custom max tokens.
     *
     * @param modelDir         path to the directory containing model files
     * @param defaultMaxTokens default maximum tokens to generate if not specified in request
     */
    public Phi3InferenceEngine(Path modelDir, int defaultMaxTokens) {
        this(modelDir, defaultMaxTokens, "cpu");
    }

    /**
     * Full constructor with backend selection.
     *
     * @param modelDir         path to the directory containing model files
     * @param defaultMaxTokens default maximum tokens to generate if not specified in request
     * @param backend          "directml", "cpu", or "auto"
     */
    public Phi3InferenceEngine(Path modelDir, int defaultMaxTokens, String backend) {
        this.modelDir = modelDir;
        this.defaultMaxTokens = defaultMaxTokens;
        this.backend = backend != null ? backend : "cpu";
    }

    /**
     * Create from {@link InferenceConfiguration}.
     * The modelPath should point to the directory containing the Phi-3 model files.
     */
    public Phi3InferenceEngine(InferenceConfiguration config) {
        this(Path.of(config.getModelPath()),
                config.getMaxTokens() > 0 ? config.getMaxTokens() : 256,
                config.getBackend());
    }

    @Override
    public void initialize() throws InferenceException {
        log.info("Phi3InferenceEngine initializing from {} (backend={})", modelDir, backend);

        try {
            long t0 = System.currentTimeMillis();

            // Load config
            Path configPath = modelDir.resolve("config.json");
            config = Phi3Config.load(configPath);
            log.info("Config: hidden={}, layers={}, heads={}, vocab={}",
                    config.hiddenSize(), config.numHiddenLayers(),
                    config.numAttentionHeads(), config.vocabSize());

            // Load tokenizer
            Path tokenizerPath = modelDir.resolve("tokenizer.json");
            tokenizer = Phi3Tokenizer.load(tokenizerPath);
            log.info("Tokenizer loaded: vocabSize={}", tokenizer.vocabSize());

            // Load weights (memory-maps the 2.1GB external data file)
            weights = Phi3Weights.load(modelDir, config);
            log.info("Weights loaded in {} ms", System.currentTimeMillis() - t0);

            // ── GPU acceleration ──────────────────────────────────────
            if (!"cpu".equalsIgnoreCase(backend) && WindowsBindings.isSupported()) {
                try {
                    wb = new WindowsBindings();
                    wb.init(backend);
                    if (wb.hasDirectMl()) {
                        int gpuLayers = Integer.getInteger("phi3.gpu.layers",
                                config.numHiddenLayers());
                        boolean gpuLmHead = Boolean.parseBoolean(
                                System.getProperty("phi3.gpu.lmhead", "true"));
                        gpuKernels = Phi3GpuKernels.create(
                                wb, weights, config, gpuLayers, gpuLmHead);
                        // V2.0: Create shared GPU pipeline
                        gpuPipeline = new Phi3GpuPipeline(wb, gpuKernels, config);
                        gpuPipeline.uploadLayerWeights(wb, weights, config);
                        log.info("GPU acceleration: {}/{} layers on GPU, lmHead={}, pipeline=V2.0",
                                gpuKernels.getGpuLayers(), config.numHiddenLayers(),
                                gpuKernels.hasLmHead());
                    } else {
                        log.warn("DirectML device not available, falling back to CPU");
                    }
                } catch (Exception e) {
                    if ("auto".equalsIgnoreCase(backend)) {
                        log.warn("GPU initialization failed, falling back to CPU: {}",
                                e.getMessage());
                        cleanupGpu();
                    } else {
                        cleanupGpu();
                        throw new InferenceException(
                                "GPU initialization failed: " + e.getMessage(), e);
                    }
                }
            }

            // Create runtime (GPU pipeline may be null → fall back to kernels or CPU)
            runtime = new Phi3Runtime(config, weights, tokenizer, gpuKernels, gpuPipeline);

            ready = true;
            log.info("Phi3InferenceEngine ready ({}ms total, backend={})",
                    System.currentTimeMillis() - t0,
                    gpuKernels != null ? "GPU(" + gpuKernels.getGpuLayers() + " layers)" : "CPU");

        } catch (IOException e) {
            throw new InferenceException("Failed to load Phi-3 model from " + modelDir, e);
        }
    }

    @Override
    public InferenceResult generate(InferenceRequest request) throws InferenceException {
        if (!ready) throw new InferenceException("Engine not initialized");

        try {
            // Format the prompt using Phi-3 chat template
            String systemPrompt = request.getSystemPrompt();
            String userPrompt = request.getUserPrompt();
            String formattedPrompt = tokenizer.formatChat(
                    systemPrompt.isBlank() ? null : systemPrompt,
                    userPrompt
            );

            int maxTokens = request.getMaxTokens() > 0 ? request.getMaxTokens() : defaultMaxTokens;
            log.info("Generating: maxTokens={}, promptLen={}", maxTokens, formattedPrompt.length());

            long t0 = System.currentTimeMillis();
            String generatedText = runtime.generate(formattedPrompt, maxTokens);
            long elapsed = System.currentTimeMillis() - t0;

            // Count tokens for usage
            int promptTokens = tokenizer.encode(formattedPrompt).length;
            int completionTokens = tokenizer.encode(generatedText).length;

            log.info("Generated {} chars ({} tokens) in {} ms ({} ms/token)",
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
            throw new InferenceException("Phi-3 generation failed: " + e.getMessage(), e);
        }
    }

    @Override
    public void shutdown() {
        ready = false;
        cleanupGpu();
        if (weights != null) {
            try {
                weights.close();
            } catch (Exception e) {
                log.warn("Error closing Phi3Weights: {}", e.getMessage());
            }
            weights = null;
        }
        runtime = null;
        tokenizer = null;
        config = null;
        log.info("Phi3InferenceEngine shut down");
    }

    /** Clean up GPU resources (idempotent). */
    private void cleanupGpu() {
        if (gpuPipeline != null) {
            try { gpuPipeline.close(); } catch (Exception e) {
                log.warn("Error closing GPU pipeline: {}", e.getMessage());
            }
            gpuPipeline = null;
        }
        if (gpuKernels != null) {
            try { gpuKernels.close(); } catch (Exception e) {
                log.warn("Error closing GPU kernels: {}", e.getMessage());
            }
            gpuKernels = null;
        }
        if (wb != null) {
            try { wb.close(); } catch (Exception e) {
                log.warn("Error closing WindowsBindings: {}", e.getMessage());
            }
            wb = null;
        }
    }

    @Override
    public boolean isReady() {
        return ready;
    }

    /**
     * Check whether a directory contains a valid Phi-3 model.
     * Looks for the 4 required files.
     */
    public static boolean isValidModelDir(Path dir) {
        return dir != null
                && Files.isDirectory(dir)
                && Files.exists(dir.resolve("config.json"))
                && Files.exists(dir.resolve("tokenizer.json"))
                && Files.exists(dir.resolve("model.onnx"))
                && Files.exists(dir.resolve("model.onnx.data"));
    }
}

