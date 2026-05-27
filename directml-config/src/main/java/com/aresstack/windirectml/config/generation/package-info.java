/**
 * Shared text-generation API and model registry.
 *
 * <h2>Model taxonomy</h2>
 *
 * <p>This project classifies models into distinct categories:
 *
 * <table>
 *   <tr><th>Category</th><th>Registry</th><th>Description</th></tr>
 *   <tr>
 *     <td><b>Embeddings</b></td>
 *     <td>{@link com.aresstack.windirectml.config.models.EmbeddingModelRegistry}</td>
 *     <td>Encoder models that map text to fixed-size vectors (MiniLM, E5, Jina).
 *         Used by the {@code embed} endpoint.</td>
 *   </tr>
 *   <tr>
 *     <td><b>Rerankers</b></td>
 *     <td>{@link com.aresstack.windirectml.config.models.EmbeddingModelRegistry}</td>
 *     <td>Cross-encoder models that score query-document relevance.
 *         Used by the {@code rerank} endpoint.</td>
 *   </tr>
 *   <tr>
 *     <td><b>Decoder-only generation (Causal LM)</b></td>
 *     <td>{@link com.aresstack.windirectml.config.generation.GenerationModelRegistry}</td>
 *     <td>Autoregressive models that generate text token-by-token
 *         (Phi-3, Qwen, Llama, GPT). Implement
 *         {@link com.aresstack.windirectml.config.generation.CausalLanguageModel}.</td>
 *   </tr>
 *   <tr>
 *     <td><b>Seq2Seq generation</b></td>
 *     <td>{@link com.aresstack.windirectml.config.generation.GenerationModelRegistry}</td>
 *     <td>Encoder-decoder models (T5, BART) that encode input fully before
 *         generating output. Future; see issue #95. Will implement
 *         {@link com.aresstack.windirectml.config.generation.TextGenerationModel}
 *         without the {@code CausalLanguageModel} marker.</td>
 *   </tr>
 *   <tr>
 *     <td><b>Summarizer (use-case adapter)</b></td>
 *     <td><em>Not a model category</em></td>
 *     <td>Application-layer adapter that wraps a generation model with a
 *         summarization system prompt and stop policy. Not a distinct model
 *         type in the registry.</td>
 *   </tr>
 * </table>
 *
 * <h2>API shape</h2>
 *
 * <ul>
 *   <li>{@link com.aresstack.windirectml.config.generation.TextGenerationModel}
 *       &ndash; public interface for any text-generation backend.</li>
 *   <li>{@link com.aresstack.windirectml.config.generation.CausalLanguageModel}
 *       &ndash; marker for autoregressive decoder-only models.</li>
 *   <li>{@link com.aresstack.windirectml.config.generation.GenerationRequest}
 *       &ndash; model-agnostic generation parameters.</li>
 *   <li>{@link com.aresstack.windirectml.config.generation.GenerationResult}
 *       &ndash; text output with token counts and timing.</li>
 *   <li>{@link com.aresstack.windirectml.config.generation.GenerationModelId}
 *       &ndash; type-safe model identifier.</li>
 *   <li>{@link com.aresstack.windirectml.config.generation.SamplerConfig}
 *       &ndash; sampling strategy (greedy, temperature, top-k).</li>
 *   <li>{@link com.aresstack.windirectml.config.generation.ChatTemplate}
 *       &ndash; prompt formatting convention per model family.</li>
 *   <li>{@link com.aresstack.windirectml.config.generation.StopTokenPolicy}
 *       &ndash; stop conditions beyond EOS.</li>
 *   <li>{@link com.aresstack.windirectml.config.generation.GenerationModelRegistry}
 *       &ndash; registry of known generation checkpoints.</li>
 * </ul>
 *
 * <h2>Design principles</h2>
 *
 * <ul>
 *   <li>Java-8 compatible &ndash; consumable by the Swing workbench.</li>
 *   <li>No Phi-3-specific types leak into the shared API.</li>
 *   <li>No embedding/reranker internals exposed.</li>
 *   <li>Qwen entries are metadata-only until the runtime is implemented.</li>
 *   <li>The registry separates generation models from the embedding registry.</li>
 * </ul>
 */
package com.aresstack.windirectml.config.generation;
