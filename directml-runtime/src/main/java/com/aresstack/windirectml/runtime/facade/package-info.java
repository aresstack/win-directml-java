/**
 * Public Java 21 facade for direct in-process use of the win-directml-java
 * local ML runtime.
 * <p>
 * This package provides a stable, high-level API surface for Java 21
 * applications that want to use embeddings, batch embeddings, and
 * reranking <b>without</b> starting the JSON-RPC sidecar process.
 *
 * <h2>Entry point</h2>
 * Use {@link com.aresstack.windirectml.runtime.facade.LocalMlRuntime#create()}
 * to obtain a runtime instance, then load models via
 * {@link com.aresstack.windirectml.runtime.facade.LocalMlRuntime#loadEmbeddingModel}
 * and {@link com.aresstack.windirectml.runtime.facade.LocalMlRuntime#loadRerankerModel}.
 *
 * <h2>Thread safety</h2>
 * The runtime itself is stateless and thread-safe. Individual model handles
 * ({@link com.aresstack.windirectml.runtime.facade.LocalEmbeddingModel},
 * {@link com.aresstack.windirectml.runtime.facade.LocalRerankerModel})
 * inherit the thread-safety characteristics of the underlying encoder/reranker
 * implementations (the shipped backends are thread-safe).
 *
 * @see com.aresstack.windirectml.runtime.facade.LocalMlRuntime
 */
package com.aresstack.windirectml.runtime.facade;
