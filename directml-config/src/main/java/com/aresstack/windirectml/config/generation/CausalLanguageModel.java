package com.aresstack.windirectml.config.generation;

/**
 * Marker interface for autoregressive decoder-only language models.
 *
 * <p>Causal (left-to-right) models generate one token at a time,
 * attending only to previously generated tokens. Examples: Phi-3, Qwen,
 * Llama, GPT-style decoders.
 *
 * <p>This marker distinguishes causal LMs from Seq2Seq models that
 * encode the full input before generating (e.g. T5, BART). Both families
 * implement {@link TextGenerationModel}, but downstream code may check
 * {@code instanceof CausalLanguageModel} to enable causal-specific
 * features (KV-cache management, speculative decoding, &hellip;).
 *
 * <p>Java-8 compatible.
 */
public interface CausalLanguageModel extends TextGenerationModel {
    // Marker interface – no additional methods beyond TextGenerationModel.
}
