package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogBackend;
import com.aresstack.windirectml.catalog.CatalogModelFamily;

/**
 * An open, package-backed generation model. Obtain one from
 * {@link GenerationRuntime#load(com.aresstack.windirectml.catalog.LocalRuntimeModelDescriptor,
 * java.nio.file.Path, CatalogBackend, LoadPolicy)}.
 *
 * <p>A handle owns native/runtime resources (device, weights, KV cache) and MUST be closed. It is
 * not guaranteed thread-safe: serialize calls, or use one handle per thread. Generation is greedy
 * and deterministic.
 */
public interface GenerationModelHandle extends AutoCloseable {

    /** Run generation to completion and return the assembled result. */
    GenerationResult generate(GenerationRequest request);

    /**
     * Run generation, delivering each token to {@code listener} as it is produced, then
     * {@link GenerationTokenListener#onComplete(GenerationResult)}.
     */
    GenerationResult generate(GenerationRequest request, GenerationTokenListener listener);

    /** The backend this handle is running on (as resolved from the requested backend). */
    CatalogBackend backend();

    /** The catalog runtime model id this handle serves. */
    String runtimeModelId();

    /** The catalog family this handle belongs to. */
    CatalogModelFamily family();

    /** Release all runtime resources. Idempotent. */
    @Override
    void close();
}
