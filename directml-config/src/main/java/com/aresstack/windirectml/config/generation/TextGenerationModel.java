package com.aresstack.windirectml.config.generation;

/**
 * Public use-case API for text generation.
 *
 * <p>Abstracts over different decoder families (Phi-3, Qwen, future models)
 * so callers can request text generation without binding to a specific runtime.
 * The generation runtime is intentionally separate from the embedding runtime:
 * embeddings have no KV cache, no chat template, no streaming &ndash; just
 * {@code text &rarr; fixed-size vector}.
 *
 * <p>Implementations:
 * <ul>
 *   <li>{@link CausalLanguageModel} &ndash; marker for autoregressive
 *       decoder-only models (Phi-3, Qwen, Llama, &hellip;).</li>
 *   <li>Future: Seq2Seq models may implement this interface directly
 *       without the {@code CausalLanguageModel} marker.</li>
 * </ul>
 *
 * <p>Use-case adapters (e.g. a Summarizer) may wrap a
 * {@code TextGenerationModel} with a fixed system prompt and stop policy,
 * but are <em>not</em> generation models themselves.
 *
 * <p>Java-8 compatible.
 */
public interface TextGenerationModel {

    /** Unique model identifier (e.g. {@code "microsoft/Phi-3-mini-4k-instruct-onnx"}). */
    GenerationModelId modelId();

    /**
     * Generate text from the given request.
     *
     * @param request generation parameters (prompt, max tokens, sampler, stop policy).
     * @return the generation result with text, token counts, and timing.
     * @throws GenerationException if generation fails.
     */
    GenerationResult generate(GenerationRequest request) throws GenerationException;

    /** Whether the model is loaded and ready to accept generation requests. */
    boolean isReady();
}
