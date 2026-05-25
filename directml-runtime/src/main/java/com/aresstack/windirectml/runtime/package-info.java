/**
 * Public Java 21 ML runtime API for local Windows DirectML inference.
 * <p>
 * Entry point: {@link com.aresstack.windirectml.runtime.WinDirectMlRuntime}.
 * <p>
 * This package provides a stable public facade for:
 * <ul>
 *   <li>Text embeddings (single and batch)</li>
 *   <li>Cross-encoder reranking</li>
 *   <li>Backend/device selection (CPU, DirectML, auto-fallback)</li>
 *   <li>Model loading and validation with clear readiness errors</li>
 * </ul>
 * <p>
 * Java 21 applications use this API directly. The {@code directml-sidecar}
 * module acts as a JSON-RPC adapter over this same API for Java 8 clients.
 */
package com.aresstack.windirectml.runtime;
