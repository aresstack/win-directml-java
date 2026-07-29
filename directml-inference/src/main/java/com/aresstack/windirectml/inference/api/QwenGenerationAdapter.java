package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogModelFamily;
import com.aresstack.windirectml.inference.InferenceEngine;
import com.aresstack.windirectml.inference.qwen.QwenInferenceEngine;

/**
 * Catalog-driven adapter for the Qwen causal/chat family (Qwen2.5-Coder-0.5B-Instruct). Allowed
 * backends per the catalog matrix: WARP, AUTO, CPU.
 */
final class QwenGenerationAdapter extends InferenceEngineAdapter {

    @Override
    public CatalogModelFamily family() {
        return CatalogModelFamily.QWEN;
    }

    @Override
    protected InferenceEngine createEngine(GenerationModelContext context) {
        return new QwenInferenceEngine(
                context.modelDirectory(), DEFAULT_MAX_TOKENS, backendString(context));
    }
}
