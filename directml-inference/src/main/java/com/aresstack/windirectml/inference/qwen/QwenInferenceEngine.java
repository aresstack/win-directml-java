package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.InferenceEngine;
import com.aresstack.windirectml.inference.InferenceException;
import com.aresstack.windirectml.inference.InferenceRequest;
import com.aresstack.windirectml.inference.InferenceResult;
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
 * <h2>Model directory layout</h2>
 * <pre>
 * model-dir/
 *   config.json          — HuggingFace model config
 *   tokenizer.json       — HuggingFace BPE tokenizer
 *   model.onnx           — ONNX model graph (weight metadata)
 *   model.onnx.data      — External weight data (memory-mapped)
 * </pre>
 *
 * <h2>Differences from Phi-3 engine</h2>
 * <ul>
 *   <li>ChatML template instead of Phi-3 role tokens</li>
 *   <li>Qwen BPE tokenizer (byte-level, ~152k vocab) instead of SentencePiece</li>
 *   <li>CPU-only: no DirectML/GPU path in this initial implementation</li>
 *   <li>GQA attention with separate Q/KV head counts</li>
 * </ul>
 */
public class QwenInferenceEngine implements InferenceEngine {

    private static final Logger log = LoggerFactory.getLogger(QwenInferenceEngine.class);

    private final Path modelDir;
    private final int defaultMaxTokens;

    private Qwen2Config config;
    private QwenTokenizer tokenizer;
    private Qwen2Weights weights;
    private Qwen2Runtime runtime;
    private boolean ready = false;

    /**
     * Create a new Qwen inference engine.
     *
     * @param modelDir        path to the model directory
     * @param defaultMaxTokens default maximum tokens if not specified in request
     */
    public QwenInferenceEngine(Path modelDir, int defaultMaxTokens) {
        this.modelDir = modelDir;
        this.defaultMaxTokens = defaultMaxTokens > 0 ? defaultMaxTokens : 256;
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

            // Create runtime
            runtime = new Qwen2Runtime(config, weights, tokenizer);

            long elapsed = System.currentTimeMillis() - t0;
            log.info("QwenInferenceEngine initialized in {} ms", elapsed);
            ready = true;

        } catch (Exception e) {
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
            throw new InferenceException("Qwen generation failed: " + e.getMessage(), e);
        }
    }

    @Override
    public void shutdown() {
        log.info("Shutting down QwenInferenceEngine");
        ready = false;
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
