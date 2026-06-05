package com.aresstack.windirectml.inference.smollm2;

/**
 * Prepared execution session for a SmolLM2 WARP executor.
 *
 * <p>A session owns whatever future native resources a concrete executor allocates. The current default session keeps
 * all data in Java metadata only and reports unsupported execution until native kernels are provided.</p>
 */
public interface SmolLM2WarpSession extends AutoCloseable {

    SmolLM2WarpRuntimePlan plan();

    SmolLM2WarpUploadManifest uploadManifest();

    SmolLM2WarpExecutionStatus status();

    SmolLM2TokenRuntimeResult generateTokenIds(SmolLM2TokenRuntimeRequest request);

    boolean closed();

    @Override
    void close();
}
