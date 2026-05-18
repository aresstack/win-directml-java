package com.aresstack.windirectml.inference;

/**
 * Abstraction over the local inference backend.
 * <p>
 * The rest of the system consumes inference through this interface only.
 * The concrete implementation is hidden behind it – callers must not know
 * about Windows / DirectML / D3D12 details.
 * <p>
 * <b>Implementations:</b>
 * <ul>
 *   <li>{@link Phi3InferenceEngine} — Phi-3-mini-4k-instruct text generation
 *       (CPU or GPU-accelerated decode, INT4 AWQ weights, greedy sampling)</li>
 *   <li>{@link MnistDirectMlEngine} — 28×28 digit classification via DirectML</li>
 *   <li>{@link StubInferenceEngine} — Deterministic stub for testing without GPU</li>
 * </ul>
 */
public interface InferenceEngine {

    /** Initialize the engine (load model, allocate resources). */
    void initialize() throws InferenceException;

    /** Run inference and produce a result. */
    InferenceResult generate(InferenceRequest request) throws InferenceException;

    /** Release all resources. */
    void shutdown();

    /** Whether the engine is ready to accept requests. */
    boolean isReady();
}
