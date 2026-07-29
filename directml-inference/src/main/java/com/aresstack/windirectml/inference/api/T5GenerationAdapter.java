package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogModelFamily;
import com.aresstack.windirectml.inference.InferenceEngine;
import com.aresstack.windirectml.inference.t5.T5InferenceEngine;

/**
 * Catalog-driven adapter for the T5 seq2seq family (t5-small, flan-t5-small, codet5-small,
 * codet5-base-multi-sum). CodeT5 is a T5 checkpoint driven through the same engine; it is resolved
 * by catalog family, not by name fragments. Allowed backends: WARP, AUTO, CPU.
 */
final class T5GenerationAdapter extends InferenceEngineAdapter {

    @Override
    public CatalogModelFamily family() {
        return CatalogModelFamily.T5;
    }

    @Override
    protected InferenceEngine createEngine(GenerationModelContext context) {
        return new T5InferenceEngine(
                context.modelDirectory(), DEFAULT_MAX_TOKENS, backendString(context));
    }
}
