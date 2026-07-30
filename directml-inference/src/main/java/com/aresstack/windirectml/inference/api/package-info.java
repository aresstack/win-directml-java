/**
 * Public generation-runtime API for the Windows DirectML/WARP local engine.
 *
 * <p><strong>This package is the only supported public API of {@code directml-inference}.</strong>
 * AskAI's Java-21 sidecar and the Swing workbench both consume the neutral contracts declared here;
 * everything else in the module (per-family engines, kernels, compilers, tokenizers, model-source
 * and wdmlpack internals) is an implementation detail and carries <em>no</em> binary- or
 * source-compatibility guarantee.
 *
 * <h2>Shape</h2>
 * <ul>
 *   <li>{@link com.aresstack.windirectml.inference.api.GenerationRuntime} — entry point; resolves a
 *       {@link com.aresstack.windirectml.catalog.LocalRuntimeModelDescriptor} + model directory +
 *       {@link com.aresstack.windirectml.catalog.CatalogBackend} into a loaded handle.</li>
 *   <li>{@link com.aresstack.windirectml.inference.api.GenerationModelHandle} — an open,
 *       package-backed model; runs greedy/deterministic generation and streams tokens; closeable.</li>
 *   <li>{@link com.aresstack.windirectml.inference.api.GenerationRequest} /
 *       {@link com.aresstack.windirectml.inference.api.GenerationResult} /
 *       {@link com.aresstack.windirectml.inference.api.GenerationToken} /
 *       {@link com.aresstack.windirectml.inference.api.GenerationTokenListener} — the call contract.</li>
 *   <li>{@link com.aresstack.windirectml.inference.api.LoadPolicy} — package-only vs. allow-compile.</li>
 *   <li>{@link com.aresstack.windirectml.inference.api.GenerationException} /
 *       {@link com.aresstack.windirectml.inference.api.GenerationErrorCode} — typed failures, e.g.
 *       {@code UNSUPPORTED_BACKEND} when a backend outside the catalog's per-family matrix is asked
 *       for (Gemma&nbsp;+&nbsp;CPU), or {@code PACKAGE_MISSING} under package-only loading.</li>
 * </ul>
 *
 * <p>The backend type is {@link com.aresstack.windirectml.catalog.CatalogBackend} (the neutral
 * cross-module enum) rather than a duplicate API-local enum, so the runtime and AskAI's Java-8 host
 * share one backend vocabulary and one allowed-backend matrix.
 */
package com.aresstack.windirectml.inference.api;
