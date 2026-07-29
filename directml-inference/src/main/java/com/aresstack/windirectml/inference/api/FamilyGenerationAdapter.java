package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogModelFamily;

/**
 * Maps a catalog family onto a concrete engine in {@code directml-inference} and opens a
 * package-backed {@link GenerationModelHandle}. One adapter per {@link CatalogModelFamily}.
 *
 * <p>Adapters resolve the model strictly from catalog metadata ({@code runtimeFamily},
 * {@code runtimeModelId}, {@code packageLifecycleId}) — never by guessing from name fragments — and
 * are responsible only for: selecting the correct package loader, honouring
 * {@link com.aresstack.windirectml.inference.api.LoadPolicy}, applying the family's
 * tokenizer/chat-template, and producing a neutral {@link GenerationResult}. Backend admissibility
 * is already enforced by {@link GenerationRuntime} before {@link #open} is called.
 *
 * <p>This is an SPI internal to the module (package-private): it is not part of the stable
 * AskAI-facing contract and AskAI is not expected to implement it. Family adapters live in this
 * same package.
 */
interface FamilyGenerationAdapter {

    /** The catalog family this adapter serves. */
    CatalogModelFamily family();

    /**
     * Open a model handle for the given resolved context.
     *
     * @throws GenerationException with an appropriate {@link GenerationErrorCode} if the model
     *     cannot be opened (missing assets, unloadable package, blocked raw-weight fallback, …)
     */
    GenerationModelHandle open(GenerationModelContext context);
}
