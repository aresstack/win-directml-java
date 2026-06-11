package com.aresstack.windirectml.inference.smollm2;

import java.util.function.IntConsumer;

/**
 * Prepared execution session for a SmolLM2 WARP executor.
 *
 * <p>A session owns the native resources a concrete executor allocates. The native session
 * ({@link SmolLM2NativeWarpSession}) runs every dense projection on the shared decoder-only WARP kernels; other
 * sessions (probe/unsupported) only report their execution status.</p>
 */
public interface SmolLM2WarpSession extends AutoCloseable {

    SmolLM2WarpRuntimePlan plan();

    SmolLM2WarpUploadManifest uploadManifest();

    SmolLM2WarpExecutionStatus status();

    /**
     * Generate token IDs for the prepared plan.
     *
     * @param request                generation request
     * @param generatedTokenConsumer optional per-token callback invoked for every accepted (non-stop) token as it is
     *                               produced; may be {@code null}. Sessions that cannot stream incrementally (legacy
     *                               executor-backed sessions) ignore it.
     */
    SmolLM2TokenRuntimeResult generateTokenIds(SmolLM2TokenRuntimeRequest request, IntConsumer generatedTokenConsumer);

    /**
     * Convenience overload without an incremental token callback.
     */
    default SmolLM2TokenRuntimeResult generateTokenIds(SmolLM2TokenRuntimeRequest request) {
        return generateTokenIds(request, null);
    }

    boolean closed();

    @Override
    void close();
}
